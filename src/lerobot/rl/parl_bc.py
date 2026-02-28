"""
PA-RL BC algorithm for diffusion policy fine-tuning.

Actor update: sample n_samples actions from ft_diffusion → Q-rank → BC distill toward best.
Critic update: IQL Q + V trained via PARLDiffusionPolicy.critic_loss().

Delegates all network logic (shared encoder, Q/V heads, action selection) to
PARLDiffusionPolicy, so this file only owns the optimizers and update schedule.
"""

import copy
import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.modeling_parl_diffusion import (
    PARLDiffusionConfig,
    PARLDiffusionPolicy,
)
from lerobot.rl.algorithm import Algorithm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PARLBCConfig:
    # PARL policy config (sampling, IQL hyperparams, network sizes)
    parl: PARLDiffusionConfig = None  # filled in __post_init__

    # Optimizers
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    grad_clip_norm: float = 1.0

    # Gradient steps per outer iteration
    n_batch_per_itr: int = 10

    # Number of outer iterations to train critic only before actor updates begin
    n_critic_warmup_itr: int = 0

    # Number of fine-tuned denoising steps: t < num_ft_train_steps uses ft_diffusion,
    # t >= num_ft_train_steps uses the frozen pretrained UNet.
    # Must equal num_train_timesteps (all steps ft) OR freeze_image_encoder must be True.
    num_ft_train_steps: int = None  # required — no default

    # Run actor update only every N outer iterations (1 = every iteration)
    policy_update_period: int = 1

    # Target Q EMA update every N critic steps
    target_update_freq: int = 1

    # Freeze image encoder during actor updates
    freeze_image_encoder: bool = False

    def __post_init__(self):
        if self.parl is None:
            self.parl = PARLDiffusionConfig()


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
            policy_type="diffusion",
            dataset_root="path/to/dataset",
        )
        train(cfg, algorithm_cls=partial(PARLBCAlgorithm, cfg=PARLBCConfig()))
    """

    def __init__(self, policy: DiffusionPolicy, cfg: PARLBCConfig = None) -> None:
        self.cfg = cfg or PARLBCConfig()

        # Upgrade base DiffusionPolicy to PARLDiffusionPolicy if needed.
        # strict=False: diffusion weights are loaded; Q/V params are randomly initialized.
        if not isinstance(policy, PARLDiffusionPolicy):
            parl_policy = PARLDiffusionPolicy(policy.config, self.cfg.parl)
            parl_policy.load_state_dict(policy.state_dict(), strict=False)
            policy = parl_policy

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

        # pretrained_unet: forever-frozen original UNet used for t >= num_ft_train_steps.
        # Captures the initial pretrained weights before any fine-tuning begins.
        self.pretrained_unet = copy.deepcopy(policy.diffusion.unet)
        self.pretrained_unet.eval()
        for p in self.pretrained_unet.parameters():
            p.requires_grad_(False)

        # Optionally freeze image encoder (exclude from actor updates)
        if self.cfg.freeze_image_encoder and hasattr(policy.diffusion, "rgb_encoder"):
            policy.diffusion.rgb_encoder.requires_grad_(False)

        # Actor optimizer: diffusion network params that still require grad
        actor_params = [p for p in policy.diffusion.parameters() if p.requires_grad]
        # Critic optimizer: Q and V heads only (not the shared encoder)
        critic_params = list(policy.critic_q.parameters()) + list(policy.critic_v.parameters())

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.cfg.critic_lr)

        # Data iterator — persists across update() calls; reset when loader changes
        self._current_loader: DataLoader | None = None
        self._data_iter = None

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
            self._data_iter = iter(loader)
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(loader)
            batch = next(self._data_iter)
        device = next(self.policy.parameters()).device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _sync_distilling_from_ft(self) -> None:
        """Copy ft_diffusion → policy.diffusion before actor phase."""
        self.policy.diffusion.load_state_dict(self.ft_diffusion.state_dict())

    def _sync_ft_from_distilling(self) -> None:
        """Copy policy.diffusion → ft_diffusion after actor phase."""
        self.ft_diffusion.load_state_dict(self.policy.diffusion.state_dict())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update(self, loader: DataLoader, itr: int) -> dict:
        itr_info: dict = {}

        # 1. Reset actor to ft_diffusion before this iteration's distillation
        self._sync_distilling_from_ft()

        # 2. Critic phase
        for step in range(self.cfg.n_batch_per_itr):
            batch = self._next_batch(loader)
            info = self._critic_update(batch)
            itr_info.update(info)
            if step % self.cfg.target_update_freq == 0:
                self.policy.update_target_q()
        
        self.critic_optimizer.zero_grad(set_to_none=True)

        # 3. Actor phase (after critic warmup, every policy_update_period iterations)
        if itr >= self.cfg.n_critic_warmup_itr and itr % self.cfg.policy_update_period == 0:
            for _ in range(self.cfg.n_batch_per_itr):
                batch = self._next_batch(loader)
                info = self._actor_update(batch)
                itr_info.update(info)
        
        self.actor_optimizer.zero_grad(set_to_none=True)

        # 4. Push updated actor back to ft_diffusion
        self._sync_ft_from_distilling()

        return itr_info

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def _critic_update(self, batch: dict) -> dict:
        q_loss, v_loss, stats = self.policy.critic_loss(batch)
        critic_loss = q_loss + v_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.critic_q.parameters()) + list(self.policy.critic_v.parameters()),
            self.cfg.grad_clip_norm,
        )
        self.critic_optimizer.step()

        return stats

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------

    def _actor_update(self, batch: dict) -> dict:
        actor_loss, stats = self.policy.actor_loss(
            batch, self.ft_diffusion,
            pretrained_unet=self.pretrained_unet,
            num_ft_train_steps=self.cfg.num_ft_train_steps,
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_params = [p for p in self.policy.diffusion.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(actor_params, self.cfg.grad_clip_norm)
        self.actor_optimizer.step()

        return stats
