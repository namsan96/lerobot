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
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
import sys

import draccus
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.transforms import ImageTransforms
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.rl.algorithm import Algorithm, AlgorithmConfig  # noqa: F401 — re-exported for back-compat
from lerobot.rl.parl_bc import PARLBCConfig  # default algorithm; registers "parl_bc" subclass
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class FTConfig:
    # Policy
    pretrained_name_or_path: str                  # HF hub id or local path
    policy_type: str                              # "diffusion", "act", "pi0", etc.

    # Dataset — all --dataset.XX options (repo_id, root, image_transforms, drop_cameras, etc.)
    dataset: DatasetConfig

    # delta_timestamps: auto-resolved from policy config if None (ft-specific, not in DatasetConfig)
    delta_timestamps: dict | None = None

    # Training
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int | None = None
    grad_clip_norm: float = 1.0

    # Minimum frames on disk before training starts (blocking wait)
    min_frames_before_training: int = 1000

    # Drop last N frames per episode from sampling; auto-set from policy config n_action_steps if 0
    drop_n_last_frames: int = 0

    # Dataset reload: wait until this many new frames arrive before each training iteration
    env_steps_per_itr: int = 100

    # Weight push: push every N iterations, but not before weight_push_warmup_itr
    weight_push_freq_itr: int = 10
    weight_push_warmup_itr: int = 0

    # Checkpointing
    output_dir: str = "outputs/ft"
    checkpoint_freq: int = 1000

    # Algorithm config — fields map to PARLBCConfig by default (e.g. --alg.actor_lr=1e-4).
    # Override the concrete type with --alg.type=<registered_name> for other algorithms.
    alg: PARLBCConfig = field(default_factory=PARLBCConfig)

    # Policy CLI overrides (--policy.xxx, injected by __main__ before train())
    policy_cli_overrides: list[str] = field(default_factory=list)


def make_policy(cfg: FTConfig, policy_cli_overrides: list[str] | None = None) -> nn.Module:
    """Instantiate policy from pretrained checkpoint (config + weights in one shot)."""
    policy_class = get_policy_class(cfg.policy_type)
    policy = policy_class.from_pretrained(cfg.pretrained_name_or_path, cli_overrides=policy_cli_overrides or [])
    return policy


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _read_dataset_info(dataset_root: Path) -> dict:
    """Cheaply read info.json without touching parquet files."""
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def _read_total_frames(dataset_root: Path) -> int:
    return _read_dataset_info(dataset_root)["total_frames"]


def _read_total_episodes(dataset_root: Path) -> int:
    return _read_dataset_info(dataset_root)["total_episodes"]


def _make_dataloader(dataset: LeRobotDataset, cfg: FTConfig) -> DataLoader:
    import math
    import torch.utils.data
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_last_frames=cfg.drop_n_last_frames,
        shuffle=True,
    )
    # Repeat indices so one epoch covers at least num_workers * prefetch_factor batches,
    # keeping workers busy across the full training update without hitting epoch boundaries.
    indices = list(sampler)
    min_samples = cfg.num_workers * (cfg.prefetch_factor or 2) * cfg.batch_size
    n_repeats = max(1, math.ceil(min_samples / max(len(indices), 1)))
    if n_repeats > 1:
        indices = indices * n_repeats
    effective_sampler = torch.utils.data.SubsetRandomSampler(indices)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=effective_sampler,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _load_dataset(cfg: FTConfig) -> LeRobotDataset:
    from torchvision.transforms import v2
    transforms = []
    if cfg.dataset.load_image_size is not None:
        transforms.append(v2.Resize(cfg.dataset.load_image_size, antialias=True))
    if cfg.dataset.image_transforms.enable:
        transforms.append(ImageTransforms(cfg.dataset.image_transforms))
    image_transforms = v2.Compose(transforms) if transforms else None

    if cfg.dataset.image_predecode:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        from lerobot.scripts.predecode_videos import predecode_videos
        dataset_root = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id
        predecode_videos(dataset_root=dataset_root, repo_id=cfg.dataset.repo_id, num_workers=1, size=cfg.dataset.predecode_size)

    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=cfg.delta_timestamps,
        image_transforms=image_transforms,
        video_backend=cfg.dataset.video_backend,
        revision=cfg.dataset.revision,
        episodes=cfg.dataset.episodes,
        drop_cameras=cfg.dataset.drop_cameras,
        image_predecode=cfg.dataset.image_predecode,
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
    from lerobot.utils.constants import HF_LEROBOT_HOME
    dataset_root = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id

    policy = make_policy(cfg, policy_cli_overrides=cfg.policy_cli_overrides)
    policy.to(device)
    policy.train()

    latest_weights = output_dir / "latest_weights.pt"
    if latest_weights.exists():
        policy.load_state_dict(torch.load(latest_weights, map_location=device))
        logging.info(f"[FT_LEARNER] Resumed weights from {latest_weights}")

    # if cfg.drop_n_last_frames == 0:
    #     cfg.drop_n_last_frames = getattr(policy.config, "horizon", 0)

    algorithm = algorithm_cls(policy, output_dir=output_dir)

    # ---- Block until enough frames are on disk ----
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            return
        try:
            n_frames = _read_total_frames(dataset_root)
        except (FileNotFoundError, KeyError):
            n_frames = 0
        if n_frames >= cfg.min_frames_before_training:
            break
        logging.info(
            f"[FT_LEARNER] Waiting for data — {n_frames}/{cfg.min_frames_before_training} frames"
        )
        time.sleep(5.0)

    dataset = _load_dataset(cfg)

    # Auto-resolve delta_timestamps now that the dataset is available
    if cfg.delta_timestamps is None:
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.utils.constants import OBS_PREFIX

        ds_meta = dataset.meta
        delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta) or {}

        # Add next.* keys for TD backup (n_obs_steps window ending n_action_steps ahead)
        n_obs = policy.config.n_obs_steps
        horizon = policy.config.horizon
        next_obs_indices = list(range(horizon + 1 - n_obs, horizon + 1))
        for key in list(delta_timestamps):
            if key.startswith(OBS_PREFIX):
                delta_timestamps[f"next.{key}"] = [i / ds_meta.fps for i in next_obs_indices]

        delta_timestamps["reward"] = [i / ds_meta.fps for i in range(horizon)]
        delta_timestamps["terminated"] = [horizon / ds_meta.fps]

        cfg.delta_timestamps = delta_timestamps or None
        logging.info(f"[FT_LEARNER] Auto-resolved delta_timestamps: {list(cfg.delta_timestamps or {})}")

        # Reload dataset with resolved delta_timestamps
        dataset = _load_dataset(cfg)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.pretrained_name_or_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        },
    )
    algorithm.preprocessor = preprocessor
    algorithm.postprocessor = postprocessor

    known_frames = 0
    initial_frames = _read_total_frames(dataset_root)
    dataloader = _make_dataloader(dataset, cfg)

    logging.info(
        f"[FT_LEARNER] Starting — {initial_frames} frames on disk, "
        f"batch_size={cfg.batch_size}, device={cfg.device}"
    )

    itr = 0

    while True:
        # ---- Shutdown check ----
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[FT_LEARNER] Shutdown requested, exiting.")
            break

        # ---- Wait until env_steps_per_itr new frames have arrived ----
        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                break
            current_frames = _read_total_frames(dataset_root)
            new_frames = current_frames - known_frames
            if new_frames >= cfg.env_steps_per_itr:
                break
            logging.info(
                f"[FT_LEARNER] Waiting for env steps — {new_frames}/{cfg.env_steps_per_itr} new frames"
            )
            time.sleep(1.0)

        known_frames = current_frames
        dataset = _load_dataset(cfg)
        dataloader = _make_dataloader(dataset, cfg)
        logging.info(f"[FT_LEARNER] Dataset reloaded — {len(dataset)} frames total ({new_frames} new)")

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

    Example (fine-tune all 100 denoising steps, freeze image encoder):

        python -m lerobot.rl.ft_learner \\
            --pretrained_name_or_path=tw_outputs/diffusion/pretrained_model \\
            --policy_type=diffusion \\
            --dataset.repo_id=user/pick_and_place \\
            --dataset.root=/data/recordings/pick_and_place \\
            --dataset.load_image_size="(224,224)" \\
            --dataset.drop_cameras="[observation.images.left]" \\
            --dataset.image_transforms.enable=true \\
            --dataset.image_transforms.tfs.rgb_shuffle.weight=1 \\
            --device=cuda \\
            --batch_size=64 \\
            --num_workers=4 \\
            --prefetch_factor=4 \\
            --min_frames_before_training=500 \\
            --env_steps_per_itr=100 \\
            --weight_push_freq_itr=5 \\
            --output_dir=outputs/rl/pick_and_place \\
            --alg.type=parl_bc \\
            --alg.num_ft_train_steps=100 \\
            --alg.actor_lr=1e-4 \\
            --alg.critic_lr=1e-4 \\
            --alg.n_batch_per_itr=10 \\
            --alg.n_critic_warmup_itr=5 \\
            --alg.freeze_image_encoder=True \\
            --policy.dinov3_hub_repo=facebookresearch/dinov2 \\
            --policy.dinov3_hub_weights=dinov2_vits14
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    root.addHandler(_h)

    if getattr(serve, "_policy_cli_overrides", None):
        cfg.policy_cli_overrides = serve._policy_cli_overrides

    train(cfg, algorithm_cls=cfg.alg.make_algorithm)


if __name__ == "__main__":
    import sys

    from lerobot.configs import parser as config_parser

    serve._policy_cli_overrides = config_parser.get_cli_overrides("policy") or []
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith("--policy.")]
    serve()
