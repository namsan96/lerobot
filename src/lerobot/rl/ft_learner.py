"""
Fine-tuning learner for online RL with async inference.

Design:
- No in-memory buffer: samples directly from LeRobotDataset on disk via DataLoader
- Periodic dataset re-instantiation to pick up new episodes from rtc_client
- Algorithm-agnostic: policy update logic lives in Algorithm subclasses
- Pushes updated weights to a file that rtc_server can watch and hot-swap
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Event

import draccus
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import get_policy_class
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class FTConfig:
    # Policy
    pretrained_name_or_path: str                  # HF hub id or local path
    policy_type: str                              # "diffusion", "act", "pi0", etc.

    # Dataset
    dataset_root: str
    dataset_repo_id: str | None = None            # repo_id for LeRobotDataset; derived from dataset_root if None
    delta_timestamps: dict | None = None          # auto-resolved from policy config if None

    # Training
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    grad_clip_norm: float = 1.0

    # Minimum frames on disk before training starts (blocking wait)
    min_frames_before_training: int = 1000

    # Drop last N frames per episode from sampling; auto-set from policy config n_action_steps if 0
    drop_n_last_frames: int = 0

    # Dataset reload: re-instantiate after this many new episodes arrive
    reload_every_n_episodes: int = 1

    # Weight push: push every N iterations, but not before weight_push_warmup_itr
    weight_push_freq_itr: int = 10
    weight_push_warmup_itr: int = 0

    # Checkpointing
    output_dir: str = "outputs/ft"
    checkpoint_freq: int = 1000

    # Policy CLI overrides (--policy.xxx, injected by __main__ before train())
    policy_cli_overrides: list[str] = field(default_factory=list)


def make_policy(cfg: FTConfig, policy_cli_overrides: list[str] | None = None) -> nn.Module:
    """Instantiate policy from pretrained checkpoint (config + weights in one shot)."""
    policy_class = get_policy_class(cfg.policy_type)
    policy = policy_class.from_pretrained(cfg.pretrained_name_or_path, cli_overrides=policy_cli_overrides or [])
    return policy


class Algorithm(ABC):
    """
    Owns the optimizer(s) for a given policy. Responsible for one full outer iteration.

    Subclasses implement different RL / IL algorithms (BC, SAC, TD3, REINFORCE, ...).
    The policy is created externally via make_policy() and passed in, so the same
    policy instance is shared between the Algorithm and the training loop (for weight push).

    The algorithm owns its own data iterator and pulls as many batches as it needs per call.
    """

    def __init__(self, policy: nn.Module) -> None:
        self._policy = policy

    @property
    def policy(self) -> nn.Module:
        return self._policy

    @abstractmethod
    def update(self, loader: DataLoader, itr: int) -> dict:
        """
        Run one full outer iteration (however many gradient steps the algorithm needs).

        Args:
            loader: DataLoader for the current dataset (may change across calls on reload).
            itr: Outer iteration index.

        Returns:
            Info dict to be logged.
        """
        ...


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _read_total_episodes(dataset_root: Path) -> int:
    """Cheaply read episode count from info.json without touching parquet files."""
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)["total_episodes"]


def _make_dataloader(dataset: LeRobotDataset, cfg: FTConfig) -> DataLoader:
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_last_frames=cfg.drop_n_last_frames,
        shuffle=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
    )


def _dataset_repo_id(cfg: FTConfig) -> str:
    """Derive repo_id from dataset_root if not explicitly set."""
    if cfg.dataset_repo_id:
        return cfg.dataset_repo_id
    parts = Path(cfg.dataset_root).parts
    return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]


def _load_dataset(cfg: FTConfig) -> LeRobotDataset:
    return LeRobotDataset(
        _dataset_repo_id(cfg),
        root=Path(cfg.dataset_root),
        delta_timestamps=cfg.delta_timestamps,
    )


# ---------------------------------------------------------------------------
# Weight push
# ---------------------------------------------------------------------------

def push_weights(policy: nn.Module, output_dir: Path) -> None:
    """
    Atomically write policy state dict to disk.

    rtc_server can watch for `latest_weights.pt` and hot-swap on each update.
    Atomic rename ensures the server never reads a partially written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "latest_weights.pt.tmp"
    dst = output_dir / "latest_weights.pt"
    torch.save(policy.state_dict(), tmp)
    tmp.rename(dst)
    logging.info(f"[FT_LEARNER] Weights pushed → {dst}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cfg: FTConfig,
    algorithm_cls: type[Algorithm],
    shutdown_event: Event | None = None,
) -> None:
    """
    Main fine-tuning loop.

    Policy is instantiated here from cfg, then passed to algorithm_cls so both
    the training loop (weight push) and the algorithm (optimizer) share the same instance.

    Args:
        cfg: FTConfig with pretrained path, dataset path, training hyperparams, etc.
        algorithm_cls: Algorithm subclass (not instance) — instantiated here with the policy.
        shutdown_event: Set this to cleanly stop the loop from another thread.
    """
    device = get_safe_torch_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    dataset_root = Path(cfg.dataset_root)

    policy = make_policy(cfg, policy_cli_overrides=cfg.policy_cli_overrides)
    policy.to(device)
    policy.train()

    # Auto-resolve delta_timestamps and drop_n_last_frames from policy config
    if cfg.delta_timestamps is None:
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.utils.constants import OBS_PREFIX

        ds_meta = LeRobotDatasetMetadata(_dataset_repo_id(cfg), root=dataset_root)
        delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta) or {}

        # Add next.* keys for TD backup (n_obs_steps window ending n_action_steps ahead)
        pcfg = policy.config
        n_obs = getattr(pcfg, "n_obs_steps", None)
        horizon = getattr(pcfg, "horizon", None)
        if n_obs is not None and horizon is not None:
            next_indices = list(range(horizon + 1 - n_obs, horizon + 1))
            for key in list(delta_timestamps):
                if key.startswith(OBS_PREFIX):
                    delta_timestamps[f"next.{key}"] = [i / ds_meta.fps for i in next_indices]

        cfg.delta_timestamps = delta_timestamps or None
        logging.info(f"[FT_LEARNER] Auto-resolved delta_timestamps: {list(cfg.delta_timestamps or {})}")

    if cfg.drop_n_last_frames == 0:
        cfg.drop_n_last_frames = getattr(policy.config, "horizon", 0)

    algorithm = algorithm_cls(policy)

    # ---- Block until enough frames are on disk ----
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            return
        known_episodes = _read_total_episodes(dataset_root)
        dataset = _load_dataset(cfg)
        if len(dataset) >= cfg.min_frames_before_training:
            break
        logging.info(
            f"[FT_LEARNER] Waiting for data — {len(dataset)}/{cfg.min_frames_before_training} frames"
        )
        time.sleep(5.0)

    dataloader = _make_dataloader(dataset, cfg)

    logging.info(
        f"[FT_LEARNER] Starting — {known_episodes} episodes, {len(dataset)} frames, "
        f"batch_size={cfg.batch_size}, device={cfg.device}"
    )

    itr = 0

    while True:
        # ---- Shutdown check ----
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[FT_LEARNER] Shutdown requested, exiting.")
            break

        # ---- Reload dataset if new episodes arrived ----
        current_episodes = _read_total_episodes(dataset_root)
        if current_episodes - known_episodes >= cfg.reload_every_n_episodes:
            logging.info(
                f"[FT_LEARNER] {current_episodes - known_episodes} new episode(s) detected "
                f"({known_episodes} → {current_episodes}), reloading dataset"
            )
            known_episodes = current_episodes
            dataset = _load_dataset(cfg)
            dataloader = _make_dataloader(dataset, cfg)
            logging.info(f"[FT_LEARNER] Dataset reloaded — {len(dataset)} frames total")

        # ---- One outer iteration (algorithm decides how many gradient steps) ----
        itr_info = algorithm.update(dataloader, itr)
        itr += 1

        logging.info(f"[FT_LEARNER] itr={itr} " + " ".join(f"{k}={v:.4f}" for k, v in itr_info.items()))

        # ---- Periodic weight push to rtc_server ----
        if itr >= cfg.weight_push_warmup_itr and itr % cfg.weight_push_freq_itr == 0:
            push_weights(algorithm.policy, output_dir)

        # ---- Checkpoint ----
        if itr % cfg.checkpoint_freq == 0:
            ckpt_path = output_dir / "checkpoints" / f"itr_{itr:08d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"itr": itr, "policy": algorithm.policy.state_dict()}, ckpt_path)
            logging.info(f"[FT_LEARNER] Checkpoint saved → {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint (PARL-BC)
# ---------------------------------------------------------------------------

@draccus.wrap()
def serve(cfg: FTConfig):
    """
    CLI entrypoint for PARL-BC fine-tuning.

    Example:
        python -m lerobot.rl.ft_learner \\
            --policy_type=diffusion \\
            --pretrained_name_or_path=tw_outputs/.../pretrained_model \\
            --dataset_root=/data/recordings/pick_and_place \\
            --device=cuda \\
            --output_dir=outputs/rl/pick_and_place \\
            --policy.dinov3_hub_repo=... \\
            --policy.dinov3_hub_weights=...
    """
    from lerobot.rl.parl_bc import PARLBCAlgorithm, PARLBCConfig

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    if getattr(serve, "_policy_cli_overrides", None):
        cfg.policy_cli_overrides = serve._policy_cli_overrides

    train(cfg, algorithm_cls=partial(PARLBCAlgorithm, cfg=PARLBCConfig()))


if __name__ == "__main__":
    import sys

    from lerobot.configs import parser as config_parser

    serve._policy_cli_overrides = config_parser.get_cli_overrides("policy") or []
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith("--policy.")]
    serve()
