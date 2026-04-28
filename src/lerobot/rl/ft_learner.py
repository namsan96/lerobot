"""
Fine-tuning learner for online RL with async inference.

Design:
- No in-memory buffer: samples directly from LeRobotDataset on disk via DataLoader
- Single-shot: load dataset, run one update, save weights, exit
- Algorithm-agnostic: policy update logic lives in Algorithm subclasses
- Saves updated weights to a named file; always symlinks latest.pt, optionally latest_weights.pt
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

    # Save name for this run's weights (e.g. "iter3" → output_dir/iter3.pt)
    save_name: str = ""

    # delta_timestamps: auto-resolved from policy config if None (ft-specific, not in DatasetConfig)
    delta_timestamps: dict | None = None

    # Training
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int | None = None
    grad_clip_norm: float = 1.0

    # Drop last N frames per episode from sampling; auto-set from policy config n_action_steps if 0
    drop_n_last_frames: int = 0

    # Whether to run critic / policy update phases
    update_critic: bool = True
    update_policy: bool = True

    # If True, also update latest_weights.pt (watched by rtc_server for hot-swap).
    # latest.pt is always updated regardless.
    update_latest_weights: bool = False

    # If True, save Q/V debug diagnostics after the update.
    debug_info: bool = True

    # Checkpointing
    output_dir: str = "outputs/ft"

    # Name (without .pt) of the checkpoint to load Q/V/actor weights from before training.
    # Special value "pretrained" skips checkpoint loading and starts from the pretrained policy as-is.
    # E.g. "latest" loads output_dir/latest.pt, "iter2" loads output_dir/iter2.pt.
    # Required — no default; use "pretrained" for the very first RL iteration.
    weight_starts_from: str = ""

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
# Weight saving
# ---------------------------------------------------------------------------

def _update_symlink(symlink: Path, target_name: str) -> None:
    """Replace (or create) a relative symlink atomically."""
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(target_name)


def save_weights(
    policy: nn.Module,
    output_dir: Path,
    save_name: str,
    update_latest_weights: bool = False,
    total_frames: int | None = None,
) -> None:
    """
    Save policy state dict to output_dir/save_name.pt.

    The checkpoint is a dict {"weights": state_dict, "total_frames": N} so metadata
    travels with the file.  Loading code extracts ["weights"] before load_state_dict.

    Always updates the latest.pt symlink (used to resume training).
    Also updates latest_weights.pt (watched by rtc_server) only when update_latest_weights=True.

    Raises FileExistsError if save_name.pt already exists.
    Symlinks are relative so the directory can be moved without breaking them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{save_name}.pt"
    if dst.exists():
        raise FileExistsError(f"Save file already exists: {dst}. Choose a different --save_name.")
    ckpt = {"weights": policy.state_dict(), "total_frames": total_frames}
    torch.save(ckpt, dst)
    _update_symlink(output_dir / "latest.pt", f"{save_name}.pt")
    msg = f"[FT_LEARNER] Weights saved → {dst}, latest.pt updated"
    if update_latest_weights:
        _update_symlink(output_dir / "latest_weights.pt", f"{save_name}.pt")
        msg += ", latest_weights.pt updated"
    logging.info(msg)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cfg: FTConfig,
    algorithm_cls: type[Algorithm],
) -> None:
    """
    Single-shot fine-tuning update.

    Loads the dataset once, runs one algorithm update, saves weights, and exits.

    Args:
        cfg: FTConfig with pretrained path, dataset path, training hyperparams, etc.
        algorithm_cls: Algorithm subclass (not instance) — instantiated here with the policy.
    """
    if not cfg.save_name:
        raise ValueError("--save_name must be set (e.g. --save_name=iter0).")

    device = get_safe_torch_device(cfg.device)
    output_dir = Path(cfg.output_dir)

    policy = make_policy(cfg, policy_cli_overrides=cfg.policy_cli_overrides)
    policy.to(device)
    policy.train()

    # Load Q/V/actor weights from the specified checkpoint.
    # Use --weight_starts_from=pretrained to skip and start directly from the pretrained policy.
    if not cfg.weight_starts_from:
        raise ValueError(
            "--weight_starts_from must be set (e.g. --weight_starts_from=pretrained, --weight_starts_from=latest, or --weight_starts_from=iter0)."
        )
    if cfg.weight_starts_from == "pretrained":
        logging.info("[FT_LEARNER] weight_starts_from=pretrained — skipping checkpoint load, using pretrained policy weights as-is.")
    else:
        weight_path = output_dir / f"{cfg.weight_starts_from}.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"weight_starts_from checkpoint not found: {weight_path}")
        ckpt = torch.load(weight_path, map_location=device)
        policy.load_state_dict(ckpt["weights"] if isinstance(ckpt, dict) else ckpt)
        logging.info(f"[FT_LEARNER] Loaded weights from {weight_path}")

    algorithm = algorithm_cls(policy, output_dir=output_dir)

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

    # ---- Early check: fail fast if the save destination already exists ----
    dst = output_dir / f"{cfg.save_name}.pt"
    if dst.exists():
        raise FileExistsError(f"Save file already exists: {dst}. Choose a different --save_name.")

    dataloader = _make_dataloader(dataset, cfg)

    logging.info(
        f"[FT_LEARNER] Starting — {len(dataset)} frames, "
        f"batch_size={cfg.batch_size}, device={cfg.device}, "
        f"update_critic={cfg.update_critic}, update_policy={cfg.update_policy}"
    )

    # ---- Single update ----
    train_stats = algorithm.update(dataloader, update_critic=cfg.update_critic, update_policy=cfg.update_policy)

    last = {**(train_stats["critic"][-1] if train_stats["critic"] else {}), **(train_stats["actor"][-1] if train_stats["actor"] else {})}
    logging.info("[FT_LEARNER] " + " ".join(f"{k}={v:.4f}" for k, v in last.items()))

    if cfg.debug_info:
        algorithm.save_debug_info(dataset, save_name=cfg.save_name, train_stats=train_stats)

    # ---- Save weights ----
    save_weights(algorithm.policy, output_dir, cfg.save_name, update_latest_weights=cfg.update_latest_weights, total_frames=len(dataset))


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
            --save_name=iter0 \\
            --output_dir=outputs/rl/pick_and_place \\
            --update_critic=true \\
            --update_policy=true \\
            --alg.type=parl_bc \\
            --alg.num_ft_train_steps=100 \\
            --alg.actor_lr=1e-4 \\
            --alg.critic_lr=1e-4 \\
            --alg.n_batch_per_itr=10 \\
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
