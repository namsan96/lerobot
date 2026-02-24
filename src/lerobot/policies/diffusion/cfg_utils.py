"""Utilities for Classifier-Free Guidance (CFG) / DPPO-style advantage-conditioned training.

Provides:
- CFGDatasetWrapper: transparent dataset wrapper that injects 'ret' or 'adv_cond' fields
- compute_returns: Monte-Carlo discounted returns from sparse binary rewards
- compute_advantages: GAE advantages using the learned value critic
"""

import numpy as np
import torch

from lerobot.utils.constants import OBS_IMAGES


class CFGDatasetWrapper(torch.utils.data.Dataset):
    """Transparent wrapper around a LeRobotDataset that injects return / advantage labels.

    Exposes `.meta`, `.num_frames`, and `.num_episodes` so that EpisodeAwareSampler
    and MetricsTracker work without changes.

    Modes:
        "base"   — pass through frames unchanged (default)
        "critic" — inject `item["ret"]` from `_rets` array
        "policy" — inject `item["adv_cond"]` from `_adv_conds` array
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.meta = dataset.meta        # for EpisodeAwareSampler
        self.num_frames = dataset.num_frames
        self.num_episodes = dataset.num_episodes
        self._rets = None        # np.ndarray (num_frames,) float32
        self._adv_conds = None   # np.ndarray (num_frames,) int64
        self.mode = "base"       # "base" | "critic" | "policy"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        if self.mode == "critic" and self._rets is not None:
            item["ret"] = torch.tensor(self._rets[idx], dtype=torch.float32)
        elif self.mode == "policy" and self._adv_conds is not None:
            item["adv_cond"] = torch.tensor(int(self._adv_conds[idx]), dtype=torch.long)
        return item


def compute_returns(dataset, gamma: float) -> np.ndarray:
    """Compute per-frame discounted returns from sparse binary rewards.

    Rewards are 0 or 1, with 1 only at the final step of a successful episode.
    Returns are computed by a standard backward discounted scan.

    Args:
        dataset: LeRobotDataset with `hf_dataset["reward"]` and `meta.episodes`.
        gamma: Discount factor.

    Returns:
        np.ndarray of shape (num_frames,) with per-frame discounted returns.
    """
    episode_from = dataset.meta.episodes["dataset_from_index"].to_numpy()
    episode_to = dataset.meta.episodes["dataset_to_index"].to_numpy()

    if "reward" in dataset.hf_dataset.column_names:
        all_rewards = np.array(dataset.hf_dataset["reward"], dtype=np.float32).reshape(-1)
    else:
        # No reward column: treat every episode as a successful demo (reward=1 at last frame).
        all_rewards = np.zeros(len(dataset), dtype=np.float32)
        all_rewards[episode_to - 1] = 1.0

    rets = np.zeros(len(dataset), dtype=np.float32)
    for from_idx, to_idx in zip(episode_from, episode_to):
        rewards = all_rewards[from_idx:to_idx]
        ret = 0.0
        for t in reversed(range(to_idx - from_idx)):
            ret = rewards[t] + gamma * ret
            rets[from_idx + t] = ret
    return rets


def compute_advantages(dataset, policy, preprocessor, gamma, gae_lambda, adv_threshold_p, device):
    """Compute per-frame GAE advantages using the learned value critic.

    Iterates over all episodes, runs the critic per frame to get V(s_t),
    then computes Generalized Advantage Estimation (GAE).

    Args:
        dataset: LeRobotDataset (unwrapped).
        policy: DiffusionPolicy with `use_critic=True` (unwrapped from accelerator).
        preprocessor: Callable that normalizes a batch and moves it to device.
        gamma: Discount factor.
        gae_lambda: GAE lambda for bias-variance trade-off.
        adv_threshold_p: Fraction of frames to label as "high advantage" (adv_cond=1).
        device: torch.device for critic inference.

    Returns:
        Tuple of:
        - all_advs: np.ndarray (num_frames,) of per-frame advantages.
        - threshold: float scalar, the quantile threshold separating high/low advantage.
    """
    episode_from = dataset.meta.episodes["dataset_from_index"].to_numpy()
    episode_to = dataset.meta.episodes["dataset_to_index"].to_numpy()

    if "reward" in dataset.hf_dataset.column_names:
        all_rewards = np.array(dataset.hf_dataset["reward"], dtype=np.float32).reshape(-1)
    else:
        all_rewards = np.zeros(len(dataset), dtype=np.float32)
        all_rewards[episode_to - 1] = 1.0

    all_advs = np.zeros(len(dataset), dtype=np.float32)

    from torch.utils.data import default_collate

    policy.eval()
    for from_idx, to_idx in zip(episode_from, episode_to):
        # Load all frames of the episode as a single batch
        frames = [dataset[i] for i in range(from_idx, to_idx)]
        batch = default_collate(frames)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch = preprocessor(batch)

        with torch.no_grad():
            if policy.config.image_features:
                batch[OBS_IMAGES] = torch.stack(
                    [batch[k] for k in policy.config.image_features], dim=-4
                )
            gc = policy.diffusion._prepare_base_global_conditioning(batch)
            values = policy.diffusion.critic(gc).squeeze(-1).cpu().numpy()  # (traj_len,)
        rewards = all_rewards[from_idx:to_idx]
        traj_len = to_idx - from_idx

        # GAE computation (standard formula)
        next_values = np.concatenate([values[1:], [values[-1]]])
        terminated = np.zeros(traj_len, dtype=bool)
        terminated[-1] = True
        deltas = rewards + (~terminated).astype(np.float32) * gamma * next_values - values

        lastgaelam = 0.0
        traj_advs = []
        for t in reversed(range(traj_len)):
            lastgaelam = deltas[t] + gamma * gae_lambda * (~terminated[t]).astype(np.float32) * lastgaelam
            traj_advs.append(lastgaelam)
        traj_advs.reverse()
        all_advs[from_idx:to_idx] = traj_advs

    threshold = float(np.quantile(all_advs, 1.0 - adv_threshold_p))
    return all_advs, threshold
