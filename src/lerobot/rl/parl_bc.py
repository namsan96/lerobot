"""
PA-RL BC algorithm for diffusion policy fine-tuning.

Actor update: sample n_samples actions from ft_diffusion → Q-rank → BC distill toward best.
Critic update: IQL Q + V trained via PARLDiffusionPolicy.critic_loss().

Delegates all network logic (shared encoder, Q/V heads, action selection) to
PARLDiffusionPolicy, so this file only owns the optimizers and update schedule.
"""

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.utils import cycle

from lerobot.policies.diffusion.modeling_parl_diffusion import (
    PARLDiffusionPolicy,
    _expand_obs,
    _stack_images,
)
from lerobot.configs.types import NormalizationMode
from lerobot.rl.algorithm import Algorithm, AlgorithmConfig
from lerobot.utils.constants import ACTION

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@AlgorithmConfig.register_subclass("parl_bc")
@dataclass
class PARLBCConfig(AlgorithmConfig):
    # Optimizers
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    grad_clip_norm: float = 1.0

    # Gradient steps per outer iteration
    n_batch_per_itr: int = 10

    # Number of fine-tuned denoising steps: t < num_ft_train_steps uses ft_diffusion,
    # t >= num_ft_train_steps uses the frozen pretrained UNet.
    # Must equal num_train_timesteps (all steps ft) OR freeze_image_encoder must be True.
    num_ft_train_steps: int = None  # required — no default

    # Target Q EMA update every N critic steps
    target_update_freq: int = 1

    # Freeze image encoder during actor updates
    freeze_image_encoder: bool = False

    # Actor algorithm: 'bc' = standard diffusion loss on best action;
    # 'chain' = reuse the sampling chain's initial noise for the flow loss (flow only).
    # 'dpo'   = DPO implicit reward loss (winner vs loser from Q ranking).
    actor_alg: str = "bc"

    # DPO hyperparameters (used when actor_alg='dpo')
    dpo_beta: float = 1.0    # KL regularisation strength
    dpo_best_k: int = 1      # winner is drawn from top-k samples
    dpo_worst_k: int = 1     # loser  is drawn from bottom-k samples
    dpo_scale: float = 1.0   # weight on DPO loss term
    bc_scale: float = 0.0    # weight on BC loss term (loss_w_ft)

    # Mixed-precision training: "fp16", "bf16", or None (disabled).
    # fp16 uses a GradScaler; bf16 does not.
    mixed_precision: str | None = None

    def make_algorithm(self, policy: "PARLDiffusionPolicy", output_dir=None) -> "PARLBCAlgorithm":
        return PARLBCAlgorithm(policy, cfg=self, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class PARLBCAlgorithm(Algorithm):
    """
    PA-RL BC: wraps PARLDiffusionPolicy with separate actor / critic optimizers.

    Per outer iteration:
        1. Sync policy.diffusion ← ft_diffusion (fresh distillation start).
        2. Critic updates: n_batch_per_itr steps; target Q EMA every target_update_freq steps.
        3. Actor updates (only if itr > n_critic_warmup_itr): n_batch_per_itr steps.
        4. Sync ft_diffusion ← policy.diffusion (reference catches up).

    Usage::

        cfg = FTConfig(
            pretrained_name_or_path="path/to/model",
            policy_type="parl_diffusion",
            dataset_root="path/to/dataset",
        )
        train(cfg, algorithm_cls=partial(PARLBCAlgorithm, cfg=PARLBCConfig()))
    """

    def __init__(self, policy: PARLDiffusionPolicy, cfg: PARLBCConfig = None, output_dir=None) -> None:
        self.cfg = cfg or PARLBCConfig()

        if not isinstance(policy, PARLDiffusionPolicy):
            raise TypeError(
                f"PARLBCAlgorithm requires a PARLDiffusionPolicy, got {type(policy).__name__}. "
                "Set policy_type='parl_diffusion' in FTConfig."
            )

        super().__init__(policy)

        if self.cfg.num_ft_train_steps is None:
            raise ValueError("PARLBCConfig.num_ft_train_steps must be set explicitly.")
        num_train_timesteps = policy.config.num_train_timesteps
        if self.cfg.num_ft_train_steps < num_train_timesteps and not self.cfg.freeze_image_encoder:
            raise ValueError(
                f"num_ft_train_steps={self.cfg.num_ft_train_steps} < num_train_timesteps={num_train_timesteps} "
                "uses a frozen pretrained UNet for high-noise steps, which limits the fine-tuning scope. "
                "freeze_image_encoder must be True in this case to prevent encoder drift."
            )

        # ft_diffusion: frozen reference diffusion network for sampling.
        # Gradients disabled; synced from policy.diffusion at end of each outer iteration.
        self.ft_diffusion = copy.deepcopy(policy.diffusion)
        self.ft_diffusion.eval()
        for p in self.ft_diffusion.parameters():
            p.requires_grad_(False)

        # pretrained_unet lives in the policy (policy.pretrained_unet), snapshotted at
        # __init__ time before any fine-tuning. Setting num_ft_train_steps on the policy
        # writes to the _num_ft_train_steps buffer so it is pushed to rtc_server with
        # the rest of the state dict.
        policy.num_ft_train_steps = self.cfg.num_ft_train_steps

        # Optionally freeze image encoder (exclude from actor updates)
        # modeling_parl_diffusion does not handle
        # 1) separated vision encoder for critics
        # 2) separated vision encoder for temporal target policy (ft_diffusion)
        assert self.cfg.freeze_image_encoder == True
        if self.cfg.freeze_image_encoder and hasattr(policy.diffusion, "rgb_encoder"):
            policy.diffusion.rgb_encoder.requires_grad_(False)
            del self.ft_diffusion.rgb_encoder  # save memory

        # Actor optimizer: diffusion network params that still require grad
        actor_params = [p for p in policy.diffusion.parameters() if p.requires_grad]
        # Critic optimizer: Q and V heads only (not the shared encoder)
        critic_params = list(policy.critic_q.parameters()) + list(policy.critic_v.parameters())

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.cfg.critic_lr)

        # Mixed-precision setup
        mp = self.cfg.mixed_precision
        if mp not in (None, "fp16", "bf16"):
            raise ValueError(f"mixed_precision must be 'fp16', 'bf16', or None, got {mp!r}")
        _dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float16)
        _device_type = next(policy.parameters()).device.type
        # autocast: enabled=False is a no-op; dtype is ignored when disabled
        self._autocast_kwargs = dict(device_type=_device_type, dtype=_dtype, enabled=mp is not None)
        # GradScaler: enabled=False makes all ops (scale/unscale/step/update) no-ops
        # bf16 is numerically stable and does not need loss scaling
        self._scaler = torch.amp.GradScaler(_device_type, enabled=(mp == "fp16"))

        # Data iterator — persists across update() calls; reset when loader changes
        self._current_loader: DataLoader | None = None
        self._data_iter = None

        self._output_dir = Path(output_dir) if output_dir is not None else Path(".")

    @property
    def policy(self) -> PARLDiffusionPolicy:
        return self._policy

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _next_batch(self, loader: DataLoader) -> dict:
        """Pull one batch from the loader, resetting the iterator on epoch end or loader change."""
        if loader is not self._current_loader:
            self._current_loader = loader
            self._data_iter = cycle(loader)
        batch = next(self._data_iter)
        if self.preprocessor is not None:
            # The preprocessor pipeline (batch_to_transition → processors → transition_to_batch)
            # only preserves obs.*, action, and next.reward/done/truncated.  Keys like "reward",
            # "terminated", and "next.observation.*" are silently dropped.  Save them first and
            # restore after preprocessing (moved to device).
            original = dict(batch)
            batch = self.preprocessor(batch)
            device = next(self.policy.parameters()).device
            for k, v in original.items():
                if k not in batch:
                    batch[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        else:
            device = next(self.policy.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _sync_distilling_from_ft(self) -> None:
        """Copy ft_diffusion → policy.diffusion before actor phase.

        rgb_encoder was deleted from ft_diffusion to save memory.
        strict=False lets those frozen keys remain untouched in policy.diffusion.
        """
        missing, unexpected = self.policy.diffusion.load_state_dict(
            self.ft_diffusion.state_dict(), strict=False
        )
        bad = [k for k in missing if not k.startswith("rgb_encoder.")]
        if bad or unexpected:
            raise RuntimeError(
                f"Sync mismatch — unexpected missing: {bad}, unexpected: {unexpected}"
            )

    def _sync_ft_from_distilling(self) -> None:
        """Copy policy.diffusion → ft_diffusion after actor phase.

        Strip rgb_encoder.* keys that were deleted from ft_diffusion.
        """
        sd = {
            k: v for k, v in self.policy.diffusion.state_dict().items()
            if not k.startswith("rgb_encoder.")
        }
        self.ft_diffusion.load_state_dict(sd, strict=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update(self, loader: DataLoader, *, update_critic: bool = True, update_policy: bool = True) -> dict:
        # 1. Reset actor to ft_diffusion before this iteration's distillation
        self._sync_distilling_from_ft()

        # 2. Critic phase
        critic_stats_list = []
        if update_critic:
            for step in range(self.cfg.n_batch_per_itr):
                batch = self._next_batch(loader)
                info = self._critic_update(batch)
                critic_stats_list.append(info)
                print(f"  critic step={step}  " + "  ".join(f"{k}={v:.6f}" for k, v in info.items()))
                if step % self.cfg.target_update_freq == 0:
                    self.policy.update_target_q()
            self.critic_optimizer.zero_grad(set_to_none=True)

        # 3. Actor phase
        actor_stats_list = []
        if update_policy:
            for step in range(self.cfg.n_batch_per_itr):
                batch = self._next_batch(loader)
                info = self._actor_update(batch)
                actor_stats_list.append(info)
                print(f"  actor  step={step}  " + "  ".join(f"{k}={v:.6f}" for k, v in info.items()))
            self.actor_optimizer.zero_grad(set_to_none=True)

        # 4. Push updated actor back to ft_diffusion
        self._sync_ft_from_distilling()

        return {"critic": critic_stats_list, "actor": actor_stats_list}

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def _critic_update(self, batch: dict) -> dict:
        self.critic_optimizer.zero_grad()
        with torch.amp.autocast(**self._autocast_kwargs):
            q_loss, v_loss, stats = self.policy.critic_loss(batch)
            critic_loss = q_loss + v_loss

        self._scaler.scale(critic_loss).backward()
        self._scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.critic_q.parameters()) + list(self.policy.critic_v.parameters()),
            self.cfg.grad_clip_norm,
        )
        self._scaler.step(self.critic_optimizer)
        self._scaler.update()

        return stats

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------

    def _actor_update(self, batch: dict) -> dict:
        self.actor_optimizer.zero_grad()
        with torch.amp.autocast(**self._autocast_kwargs):
            actor_loss, stats = self.policy.actor_loss(
                batch, self.ft_diffusion,
                pretrained_unet=self.policy.pretrained_unet,
                num_ft_train_steps=self.cfg.num_ft_train_steps,
                actor_alg=self.cfg.actor_alg,
                dpo_beta=self.cfg.dpo_beta,
                dpo_best_k=self.cfg.dpo_best_k,
                dpo_worst_k=self.cfg.dpo_worst_k,
                dpo_scale=self.cfg.dpo_scale,
                bc_scale=self.cfg.bc_scale,
            )

        self._scaler.scale(actor_loss).backward()
        self._scaler.unscale_(self.actor_optimizer)
        actor_params = [p for p in self.policy.diffusion.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(actor_params, self.cfg.grad_clip_norm)
        self._scaler.step(self.actor_optimizer)
        self._scaler.update()

        return stats

    # ------------------------------------------------------------------
    # Debug info
    # ------------------------------------------------------------------

    @torch.no_grad()
    def save_debug_info(self, dataset, save_name: str, train_stats: dict | None = None) -> None:
        """Compute and save debug diagnostics for the current policy.

        Samples up to 5 random episodes, then for each episode:
          - Computes Q(s_t, a_t) and V(s_t) at every timestep t using the
            dataset action (behaviour-policy evaluation).
          - At every horizon-th timestep, draws 20 action chunks from the
            current diffusion policy and records their Q scores.
          - Attaches the raw per-step rewards and terminated flags for the
            full episode.

        Everything is saved as a single .pt file:
            {debug_info_dir}/{save_name}.pt
        """
        policy = self.policy
        was_training = policy.training
        policy.eval()
        device = next(policy.parameters()).device

        n_ep = dataset.meta.total_episodes
        ep_indices = random.sample(range(n_ep), min(5, n_ep))

        # Ensure the underlying HF dataset is loaded (no-op if already loaded).
        dataset._ensure_hf_dataset_loaded()

        episodes_out = []
        for ep_idx in ep_indices:
            ep_meta = dataset.meta.episodes[ep_idx]
            from_idx = ep_meta["dataset_from_index"]
            to_idx = ep_meta["dataset_to_index"]
            ep_len = to_idx - from_idx

            # Raw per-step rewards and terminated flags for the whole episode.
            raw_frames = dataset.hf_dataset[list(range(from_idx, to_idx))]
            rewards_ep = torch.stack(raw_frames["reward"]).float()       # (ep_len,)
            terminated_ep = torch.stack(raw_frames["terminated"]).float()  # (ep_len,)

            # ------------------------------------------------------------------
            # Q and V at every timestep (using dataset actions)
            # ------------------------------------------------------------------
            q_vals, v_vals, actions_ep = [], [], []
            for t in range(ep_len):
                item = dataset[from_idx + t]
                batch = {
                    k: v.unsqueeze(0)
                    for k, v in item.items() if isinstance(v, torch.Tensor)
                }
                if self.preprocessor is not None:
                    batch = self.preprocessor(batch)
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}
                batch = _stack_images(batch, policy)

                encoder_tokens, state_flat, cls_tokens_flat = policy.encode_obs(batch)
                action = batch[ACTION][:, :policy._n_act]  # (1, n_act, Da)

                q_vals.append(policy.critic_q(encoder_tokens, state_flat, action).squeeze(0).cpu())
                v_vals.append(policy.critic_v(state_flat, cls_tokens_flat).squeeze(0).cpu())
                actions_ep.append(item[ACTION][0].cpu())  # (Da,) unnormalized (raw from dataset)

            # ------------------------------------------------------------------
            # Sample 20 action chunks at obs[::horizon]
            # ------------------------------------------------------------------
            sampled_at_keyframes = []
            for t in range(0, ep_len, policy._horizon):
                item = dataset[from_idx + t]
                batch = {
                    k: v.unsqueeze(0)
                    for k, v in item.items() if isinstance(v, torch.Tensor)
                }
                if self.preprocessor is not None:
                    batch = self.preprocessor(batch)
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}
                batch = _stack_images(batch, policy)

                dm = policy.diffusion
                n_samp = 20
                obs_n = _expand_obs(batch, n_samp)
                global_cond_n = dm._prepare_global_conditioning(obs_n)
                pt_unet = policy.pretrained_unet if policy.num_ft_train_steps > 0 else None
                sampled = dm.conditional_sample(
                    n_samp,
                    global_cond=global_cond_n,
                    pretrained_unet=pt_unet,
                    num_ft_train_steps=policy.num_ft_train_steps,
                    noise_injection_std=policy.config.noise_injection_std,
                ).view(
                    n_samp, 1, policy._horizon, policy._action_dim
                )  # (20, 1, H, Da)

                encoder_tokens, state_flat, _ = policy.encode_obs(batch)
                q_scores = policy._q_score_samples(encoder_tokens, state_flat, sampled)  # (20, 1)

                actions_out = sampled.squeeze(1)  # (20, H, Da)
                if self.postprocessor is not None:
                    actions_out = self.postprocessor(actions_out)
                sampled_at_keyframes.append({
                    "t": t,
                    "actions": actions_out.cpu(),            # (20, H, Da) unnormalized
                    "q_scores": q_scores.squeeze(-1).cpu(),  # (20,)
                })

            episodes_out.append({
                "ep_idx": ep_idx,
                "q": torch.stack(q_vals),           # (ep_len,)
                "v": torch.stack(v_vals),           # (ep_len,)
                "rewards": rewards_ep.cpu(),         # (ep_len,)
                "terminated": terminated_ep.cpu(),   # (ep_len,)
                "actions": torch.stack(actions_ep), # (ep_len, n_act, Da)
                "sampled_actions": sampled_at_keyframes,
            })

        out_path = self._output_dir / f"{save_name}_debug_info.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"episodes": episodes_out, "train_stats": train_stats}, out_path)
        log.info(f"[PARL] Debug info saved → {out_path}")

        if was_training:
            policy.train()
