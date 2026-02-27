"""
PA-RL Diffusion Policy: DiffusionPolicy extended with Transformer Q and MLP V critics.

Only supports use_dit=True (DINOv3 + Gr00tDiT backbone). Raises NotImplementedError otherwise.

Architecture
------------
- Actor (Gr00tDiTUnetWrapper + DINOv3): identical to DiffusionPolicy.
- Q(s, a): Plain Transformer over (state_token + action_tokens), cross-attending image
           patch tokens. Reads only the state token (index 0) after the final block →
           Linear head → scalar. No diffusion timestep; no output projection over actions.
- V(s):   MLP over concat(state_flat, DINOv3 CLS tokens).

Gradient flow
-------------
- Actor backward (BC loss via compute_loss):
    gradients flow through DiT unet AND DINOv3 rgb_encoder.
- Critic backward (IQL Q+V loss):
    encode_obs is called under torch.no_grad(); Q/V heads learn from detached features,
    so gradients never reach the shared image encoder.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from diffusers.models.attention import BasicTransformerBlock

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PARL-specific configuration
# ---------------------------------------------------------------------------

@dataclass
class PARLDiffusionConfig:
    # Sampling
    n_samples: int = 16
    distil_temp: float = 0.0        # 0 = argmax; >0 = softmax temperature selection

    # IQL critic
    expectile: float = 0.7          # asymmetric weight for V expectile regression
    gamma: float = 0.99             # discount factor for TD backup
    tau: float = 0.005              # EMA rate for target Q network

    # Q Transformer architecture (independent of policy DiT config)
    q_hidden_dim: int = 256
    q_n_heads: int = 4
    q_n_layers: int = 4
    q_dropout: float = 0.0

    # V network MLP hidden dims
    v_hidden_dims: tuple[int, ...] = (256, 256)

    # Dataset keys
    reward_key: str = "next.reward"
    done_key: str = "next.done"
    # Optional: if these keys exist in the batch, TD backup is used for Q target.
    next_obs_state_key: str = "next.observation.state"
    next_obs_image_prefix: str = "next.observation.images"

    # Inference server mode: skip critic_v and critic_target_q (Q only, no V or target Q)
    inference_only: bool = False


# ---------------------------------------------------------------------------
# Transformer Q network
# ---------------------------------------------------------------------------

class TransformerQNetwork(nn.Module):
    """
    Q(encoder_tokens, state, action) → scalar.

    Uses GR00T's BasicTransformerBlock (norm_type="layer_norm", no timestep conditioning):
      - state_encoder: Linear  state_dim → H
      - action_encoder: Linear  action_dim → H  (+ position embeddings)
      - N BasicTransformerBlocks: self-attn over [state_token | action_tokens],
        cross-attn to encoder_tokens
      - Read state token (index 0) after final LayerNorm → Linear head → scalar
    """

    def __init__(
        self,
        encoder_token_dim: int,
        state_dim: int,
        action_dim: int,
        n_action_steps: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.action_position_embedding = nn.Embedding(n_action_steps, hidden_dim)
        nn.init.normal_(self.action_position_embedding.weight, 0.0, 0.02)

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim,
                num_attention_heads=n_heads,
                attention_head_dim=hidden_dim // n_heads,
                dropout=dropout,
                cross_attention_dim=encoder_token_dim,
                activation_fn="geglu",
                norm_type="layer_norm",
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_tokens: Tensor, state: Tensor, action: Tensor) -> Tensor:
        """
        encoder_tokens : (B, N, D)           image patch tokens (cross-attention context)
        state          : (B, state_dim)       flattened obs state
        action         : (B, T, action_dim)   n_action_steps actions
        Returns        : (B,)                 Q values
        """
        B, T, _ = action.shape

        state_embed = self.state_encoder(state).unsqueeze(1)      # (B, 1, H)
        action_embed = self.action_encoder(action)                 # (B, T, H)
        pos_ids = torch.arange(T, device=action.device)
        action_embed = action_embed + self.action_position_embedding(pos_ids)
        x = torch.cat([state_embed, action_embed], dim=1)         # (B, 1+T, H)

        for block in self.blocks:
            x = block(x, encoder_hidden_states=encoder_tokens)

        state_token = self.norm(x[:, 0, :])                       # (B, H) — state token only
        return self.q_head(state_token).squeeze(-1)                # (B,)


# ---------------------------------------------------------------------------
# V network (MLP)
# ---------------------------------------------------------------------------

class VNetwork(nn.Module):
    """V(state_flat, cls_tokens_flat) → scalar. MLP over their concatenation."""

    def __init__(self, obs_feat_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_feat_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_feat: Tensor) -> Tensor:
        """obs_feat: (B, obs_feat_dim)  →  (B,)"""
        return self.net(obs_feat).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_images(batch: dict, policy: "PARLDiffusionPolicy") -> dict:
    """Stack individual camera keys into OBS_IMAGES, same as DiffusionPolicy.forward."""
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack(
            [batch[k] for k in policy.config.image_features], dim=-4
        )
    return batch


def _expand_obs(obs: dict, n: int) -> dict:
    """Repeat obs n times along the batch dim: (B, ...) → (n*B, ...)."""
    return {k: v.repeat(n, *((1,) * (v.dim() - 1))) for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class PARLDiffusionPolicy(DiffusionPolicy):
    """
    DiffusionPolicy (DiT + DINOv3) with DiT Q and MLP V critics for RL fine-tuning.
    Requires config.use_dit=True; raises NotImplementedError otherwise.

    New public methods
    ------------------
    encode_obs(batch)                  → (encoder_tokens, state_flat, cls_tokens_flat)
    critic_loss(batch)                 → (q_loss, v_loss, stats)
    actor_loss(batch, ft_diffusion)    → (loss, stats)
    update_target_q()                  EMA update of critic_target_q

    Override
    --------
    predict_action_chunk(...)       uses Q-based action selection
    from_pretrained(...)            loads in inference_only mode by default (Q only, no V/target_q)
    """

    name = "parl_diffusion"

    def __init__(self, config: DiffusionConfig, parl_cfg: PARLDiffusionConfig | None = None) -> None:
        super().__init__(config)
        self.parl_cfg = parl_cfg or PARLDiffusionConfig()

        if not config.use_dit:
            raise NotImplementedError(
                "PARLDiffusionPolicy only supports use_dit=True "
                "(Gr00tDiT backbone with DINOv3 encoder)."
            )

        self._n_act = config.n_action_steps
        self._horizon = config.horizon
        self._action_dim = config.action_feature.shape[0]
        self._n_obs_steps = config.n_obs_steps
        self._n_cams = len(config.image_features) if config.image_features else 0

        # Q/V input dims — match Gr00tDiTUnetWrapper's state_dim convention
        # (robot_state * n_obs_steps + env_state, no adv_embed)
        state_dim = (
            config.robot_state_feature.shape[0] * config.n_obs_steps
            + (config.env_state_feature.shape[0] if config.env_state_feature else 0)
        )
        encoder_token_dim = self.diffusion.rgb_encoder.feature_dim  # DINOv3 embed_dim
        cls_dim = encoder_token_dim  # CLS token has same dim as patch tokens

        pcfg = self.parl_cfg
        self.critic_q = TransformerQNetwork(
            encoder_token_dim=encoder_token_dim,
            state_dim=state_dim,
            action_dim=self._action_dim,
            n_action_steps=self._n_act,
            hidden_dim=pcfg.q_hidden_dim,
            n_heads=pcfg.q_n_heads,
            n_layers=pcfg.q_n_layers,
            dropout=pcfg.q_dropout,
        )

        if not pcfg.inference_only:
            self.critic_target_q = copy.deepcopy(self.critic_q)
            self.critic_target_q.requires_grad_(False)
            self.critic_target_q.eval()   # dropout must be off for stable targets

            v_in_dim = state_dim + cls_dim * self._n_obs_steps * self._n_cams
            self.critic_v = VNetwork(v_in_dim, pcfg.v_hidden_dims)

        # Set to True after first weight hot-swap from ft_learner.
        # Until then, predict_action_chunk falls back to plain diffusion sampling.
        self._q_initialized: bool = False

    def train(self, mode: bool = True) -> "PARLDiffusionPolicy":
        """Keep critic_target_q permanently in eval mode regardless of parent train() calls."""
        super().train(mode)
        if hasattr(self, "critic_target_q"):
            self.critic_target_q.eval()
        return self

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Load in inference_only mode by default: Q network only, no V or target Q.

        The training path (PARLBCAlgorithm) never calls from_pretrained for PARLDiffusionPolicy —
        it upgrades a loaded DiffusionPolicy directly. So from_pretrained is only called by the
        inference server, which only needs critic_q for Q-guided action selection.

        Override by passing parl_cfg=PARLDiffusionConfig(inference_only=False) explicitly.
        """
        kwargs.setdefault("parl_cfg", PARLDiffusionConfig(inference_only=True))
        return super().from_pretrained(pretrained_name_or_path, **kwargs)

    # ------------------------------------------------------------------
    # Obs encoding
    # ------------------------------------------------------------------

    def _get_patch_and_cls_tokens(self, images_flat: Tensor) -> tuple[Tensor, Tensor]:
        """
        Preprocess images and run DINOv3 backbone to get both patch and CLS tokens.

        Replicates DiffusionDinoV3Encoder preprocessing (resize + ImageNet normalization)
        then calls backbone.forward_features to extract both token types in one pass.

        images_flat : (M, C, H, W) in [0, 1]
        Returns: patch_tokens (M, N, D), cls_token (M, D)
        """
        enc = self.diffusion.rgb_encoder
        if images_flat.shape[-2:] != (enc.IMG_SIZE, enc.IMG_SIZE):
            images_flat = enc.resize(images_flat)
        mean = torch.tensor(enc.IMAGENET_MEAN, device=images_flat.device, dtype=images_flat.dtype).view(1, 3, 1, 1)
        std = torch.tensor(enc.IMAGENET_STD, device=images_flat.device, dtype=images_flat.dtype).view(1, 3, 1, 1)
        images_flat = (images_flat - mean) / std
        out = enc.backbone.forward_features(images_flat)
        return out["x_norm_patchtokens"], out["x_norm_clstoken"]

    def _encode_obs_dit(self, obs: dict) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode observations for Q and V critics in one DINOv3 pass.

        Returns:
            encoder_tokens  : (B, s*n*N, D)   patch tokens for Q DiT cross-attention
            state_flat      : (B, state_dim)   flattened state for Q (and concat to V input)
            cls_tokens_flat : (B, s*n*D)       DINOv3 CLS tokens for V
        """
        B, s = obs[OBS_STATE].shape[:2]
        n = obs[OBS_IMAGES].shape[2]

        images_flat = einops.rearrange(obs[OBS_IMAGES], "b s n ... -> (b s n) ...")
        patch_tokens, cls_tokens = self._get_patch_and_cls_tokens(images_flat)

        encoder_tokens = einops.rearrange(
            patch_tokens, "(b s n) t d -> b (s n t) d", b=B, s=s, n=n
        )
        cls_tokens_flat = einops.rearrange(
            cls_tokens, "(b s n) d -> b (s n d)", b=B, s=s, n=n
        )

        state_feats = [obs[OBS_STATE]]
        if self.config.env_state_feature:
            state_feats.append(obs[OBS_ENV_STATE])
        state_flat = torch.cat(state_feats, dim=-1).flatten(1)  # (B, state_dim)

        return encoder_tokens, state_flat, cls_tokens_flat

    def encode_obs(self, obs: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (encoder_tokens, state_flat, cls_tokens_flat) for Q and V critics."""
        return self._encode_obs_dit(obs)

    # ------------------------------------------------------------------
    # Inference: Q-guided action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        full_length: bool = False,
        action_cond=None,
    ) -> Tensor:
        """Override: sample n_samples actions, return Q-optimal one.

        Falls back to plain diffusion sampling until Q weights have been loaded
        from ft_learner (i.e. until _q_initialized is set True by the weights watcher).
        """
        if not self._q_initialized:
            return super().predict_action_chunk(batch, noise=noise, full_length=full_length, action_cond=action_cond)
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        batch = _stack_images(batch, self)
        return self._sample_and_select(batch, self.parl_cfg.n_samples, full_length, action_cond=action_cond)

    @torch.no_grad()
    def _sample_and_select(self, batch: dict, n_samples: int, full_length: bool = False, action_cond=None) -> Tensor:
        """
        Sample n_samples full-horizon actions from self.diffusion, Q-score, return best.

        batch must have OBS_IMAGES already stacked.
        action_cond: (B, T, Da) committed prefix — expanded to (n*B, T, Da) for all samples.
        """
        dm = self.diffusion
        B = batch[OBS_STATE].shape[0]

        encoder_tokens, state_flat, _ = self.encode_obs(batch)  # CLS not needed for Q scoring

        obs_n = _expand_obs(batch, n_samples)
        global_cond_n = dm._prepare_global_conditioning(obs_n)
        action_cond_n = (
            action_cond.repeat(n_samples, *((1,) * (action_cond.dim() - 1)))
            if action_cond is not None else None
        )
        sampled = dm.conditional_sample(
            n_samples * B, global_cond=global_cond_n, action_cond=action_cond_n
        ).view(n_samples, B, self._horizon, self._action_dim)

        q_scores = self._q_score_samples(encoder_tokens, state_flat, sampled)
        best = self._select_best(sampled, q_scores)

        start = dm.config.n_obs_steps - 1
        return best[:, start:] if full_length else best[:, start:start + self._n_act]

    def _q_score_samples(
        self,
        encoder_tokens: Tensor,
        state_flat: Tensor,
        sampled: Tensor,
    ) -> Tensor:
        """
        encoder_tokens : (B, N, D)
        state_flat     : (B, state_dim)
        sampled        : (n_samples, B, horizon, Da)
        Returns        : (n_samples, B)
        """
        n, B = sampled.shape[:2]
        enc_n = encoder_tokens.unsqueeze(0).expand(n, -1, -1, -1).reshape(n * B, *encoder_tokens.shape[1:])
        state_n = state_flat.unsqueeze(0).expand(n, -1, -1).reshape(n * B, -1)
        act_n = sampled[:, :, :self._n_act].reshape(n * B, self._n_act, self._action_dim)
        return self.critic_q(enc_n, state_n, act_n).view(n, B)

    def _select_best(self, sampled: Tensor, q_scores: Tensor) -> Tensor:
        """
        sampled  : (n_samples, B, horizon, Da)
        q_scores : (n_samples, B)
        returns  : (B, horizon, Da)
        """
        n, B = sampled.shape[:2]
        if self.parl_cfg.distil_temp > 0:
            probs = F.softmax(q_scores / self.parl_cfg.distil_temp, dim=0).T  # (B, n)
            best_idx = Categorical(probs=probs).sample()                       # (B,)
        else:
            best_idx = q_scores.argmax(dim=0)                                  # (B,)

        return sampled.gather(
            0,
            best_idx[None, :, None, None].expand(1, B, self._horizon, self._action_dim),
        ).squeeze(0)

    # ------------------------------------------------------------------
    # Critic loss (IQL)
    # ------------------------------------------------------------------

    def critic_loss(self, batch: dict) -> tuple[Tensor, Tensor, dict]:
        """
        IQL Q + V losses.

        Q target: r + γ * (1 - done) * V(s')   if next-obs keys are present,
                  r                             otherwise.
        V target: expectile regression against Q_target(s, a_dataset).

        encode_obs is called under torch.no_grad() so critic gradients do NOT update
        the shared image encoder.

        Returns (q_loss, v_loss, stats_dict).
        """
        pcfg = self.parl_cfg
        obs = _stack_images(batch, self)

        with torch.no_grad():
            encoder_tokens, state_flat, cls_tokens_flat = self.encode_obs(obs)

        v_feat = torch.cat([state_flat, cls_tokens_flat], dim=-1)  # (B, v_in_dim)
        action = batch[ACTION][:, :self._n_act]                    # (B, n_act, Da)
        reward = batch[pcfg.reward_key].float()                    # (B,)
        done = batch.get(pcfg.done_key, torch.zeros_like(reward)).float()

        # V loss: expectile regression against frozen target Q
        with torch.no_grad():
            q_ref = self.critic_target_q(encoder_tokens, state_flat, action)  # (B,)
        v = self.critic_v(v_feat)                                              # (B,)
        diffs = q_ref - v
        weights = torch.where(diffs > 0, pcfg.expectile, 1.0 - pcfg.expectile)
        v_loss = (weights * diffs.pow(2)).mean()

        # Q loss: TD backup with V(s')
        next_obs = self._build_next_obs_batch(batch)
        with torch.no_grad():
            next_enc_tokens, next_state_flat, next_cls_flat = self.encode_obs(next_obs)
            next_v_feat = torch.cat([next_state_flat, next_cls_flat], dim=-1)
            next_v = self.critic_v(next_v_feat)                                # (B,)
        q_target = reward + pcfg.gamma * (1.0 - done) * next_v

        q = self.critic_q(encoder_tokens, state_flat, action)                  # (B,)
        q_loss = F.mse_loss(q, q_target.detach())

        stats = {
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "q_mean": q.detach().mean().item(),
            "v_mean": v.detach().mean().item(),
        }
        return q_loss, v_loss, stats

    def _build_next_obs_batch(self, batch: dict) -> dict:
        """Build next-obs dict for encode_obs from "next.*" delta_timestamps keys.

        Expects batch["next.observation.state"] with shape (B, 1, state_dim) and,
        for image policies, batch["next.observation.images.*"] with shape (B, 1, C, H, W).
        These are produced by LeRobotDataset when delta_timestamps includes "next.*" keys
        with a delta of n_action_steps / fps.  Terminal frames must be excluded via
        drop_n_last_frames = n_action_steps in FTConfig.
        """
        pcfg = self.parl_cfg
        next_obs: dict[str, Tensor] = {OBS_STATE: batch[pcfg.next_obs_state_key]}

        next_img_keys = [k for k in batch if k.startswith(pcfg.next_obs_image_prefix + ".")]
        if self.config.image_features:
            remapped = {
                k.replace(pcfg.next_obs_image_prefix, "observation.images"): batch[k]
                for k in next_img_keys
            }
            next_obs[OBS_IMAGES] = torch.stack(
                [remapped[k] for k in self.config.image_features], dim=-4
            )
        return next_obs

    # ------------------------------------------------------------------
    # Actor loss (BC distillation toward Q-optimal sample)
    # ------------------------------------------------------------------

    def actor_loss(self, batch: dict, ft_diffusion: nn.Module) -> tuple[Tensor, dict]:
        """
        1. Sample n_samples full-horizon actions from ft_diffusion (frozen reference, no grad).
        2. Q-score on n_action_steps portion using current policy's obs encoding (no grad).
        3. Select best per state.
        4. Diffusion BC loss toward best action (gradients through self.diffusion).

        Gradients flow through unet AND rgb_encoder (encoder is trained by the actor).
        ft_diffusion is the frozen reference network; only self.diffusion is updated.
        """
        pcfg = self.parl_cfg
        n = pcfg.n_samples
        obs = _stack_images(batch, self)
        B = batch[ACTION].shape[0]
        device = next(self.parameters()).device

        # Sample from ft_diffusion and Q-score (all no grad)
        with torch.no_grad():
            encoder_tokens, state_flat, _ = self.encode_obs(obs)
            obs_n = _expand_obs(obs, n)
            global_cond_n = ft_diffusion._prepare_global_conditioning(obs_n)
            sampled = ft_diffusion.conditional_sample(                 # (n*B, H, Da)
                n * B, global_cond=global_cond_n
            ).view(n, B, self._horizon, self._action_dim)

            q_scores = self._q_score_samples(encoder_tokens, state_flat, sampled)  # (n, B)
            best_action = self._select_best(sampled, q_scores)                      # (B, H, Da)

        # BC loss — grad flows through self.diffusion.compute_loss
        #    → _prepare_global_conditioning → rgb_encoder
        bc_batch = dict(obs)
        bc_batch[ACTION] = best_action
        bc_batch["action_is_pad"] = torch.zeros(
            B, self._horizon, dtype=torch.bool, device=device
        )
        actor_loss = self.diffusion.compute_loss(bc_batch)

        stats = {
            "actor_loss": actor_loss.item(),
            "q_best_mean": q_scores.max(dim=0).values.mean().item(),
            "q_spread_mean": (
                q_scores.max(dim=0).values - q_scores.min(dim=0).values
            ).mean().item(),
        }
        return actor_loss, stats

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target_q(self) -> None:
        """EMA update: target_q ← τ * q + (1 - τ) * target_q."""
        if not hasattr(self, "critic_target_q"):
            raise RuntimeError("update_target_q called on inference_only PARLDiffusionPolicy")
        tau = self.parl_cfg.tau
        for p, tp in zip(self.critic_q.parameters(), self.critic_target_q.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
