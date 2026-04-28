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
import math
from dataclasses import dataclass, field

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from diffusers.models.attention import BasicTransformerBlock

from speedaug.models.common.mlp import MLP, ResidualMLP
from speedaug.models.common.modules import SpatialEmb

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PARL-specific configuration
# ---------------------------------------------------------------------------

@PreTrainedConfig.register_subclass("parl_diffusion")
@dataclass
class PARLDiffusionConfig(DiffusionConfig):
    # Sampling
    n_samples: int = 5
    use_q_selection: bool = True   # if False, skip multi-sample Q scoring and use a single diffusion sample
    distil_temp: float = 0.0        # 0 = argmax; >0 = softmax temperature selection
    noise_injection_std: float = 0.0  # additive noise per denoising step (flow matching)

    # IQL critic
    expectile: float = 0.5          # asymmetric weight for V expectile regression
    gamma: float = 0.99             # discount factor for TD backup
    tau: float = 0.005              # EMA rate for target Q network
    q_target_clip_min: float = float("-inf")  # lower-clamp on Q target; -inf = no-op
    q_target_clip_max: float = float("inf")  # lower-clamp on Q target; -inf = no-op

    # Q network type: "transformer" (cross-attending DiT) or "mlp" (SpatialEmb + MLP)
    q_type: str = "transformer"

    # Q Transformer architecture (independent of policy DiT config)
    q_hidden_dim: int = 128
    q_n_heads: int = 4
    q_n_layers: int = 2
    q_dropout: float = 0.0

    # MLP Q head hidden dims (used when q_type="mlp")
    q_mlp_dims: tuple[int, ...] = (1024, 1024, 1024)

    # V network MLP hidden dims
    v_hidden_dims: tuple[int, ...] = (256, 256)

    # Dataset keys
    reward_key: str = "reward"
    terminated_key: str = "terminated"
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
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
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
# MLP Q network
# ---------------------------------------------------------------------------

class MLPQNetwork(nn.Module):
    """
    Q(encoder_tokens, state, action) → scalar.

    Mirrors ViTCriticObsAct from speedaug:
      - SpatialEmb compresses encoder patch tokens (conditioned on state) → spatial_emb vector
      - Concat [spatial_emb | state | action_flat] → MLP/ResidualMLP head → scalar
    """

    def __init__(
        self,
        num_patch: int,
        encoder_token_dim: int,
        state_dim: int,
        action_dim: int,
        n_action_steps: int,
        spatial_emb: int = 128,
        mlp_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        activation_type: str = "Mish",
        use_layernorm: bool = True,
        residual_style: bool = True,
    ) -> None:
        super().__init__()
        self.compress = SpatialEmb(
            num_patch=num_patch,
            patch_dim=encoder_token_dim,
            prop_dim=state_dim,
            proj_dim=spatial_emb,
            dropout=dropout,
        )
        model = ResidualMLP if residual_style else MLP
        q_in_dim = spatial_emb + state_dim + action_dim * n_action_steps
        self.q_head = model(
            [q_in_dim] + list(mlp_dims) + [1],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, encoder_tokens: Tensor, state: Tensor, action: Tensor) -> Tensor:
        """
        encoder_tokens : (B, N, D)           patch tokens (N = n_obs_steps * n_cams * patches_per_img)
        state          : (B, state_dim)       flattened obs state
        action         : (B, T, action_dim)   n_action_steps actions
        Returns        : (B,)                 Q values
        """
        feat = self.compress(encoder_tokens, state)       # (B, spatial_emb)
        action_flat = action.view(action.shape[0], -1)    # (B, T*Da)
        x = torch.cat([feat, state, action_flat], dim=-1)
        return self.q_head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# V network (MLP)
# ---------------------------------------------------------------------------

class VNetwork(nn.Module):
    """V(state_flat, cls_token) → scalar.

    Encodes state with a 2-layer MLP (mirrors Q-network state_encoder pattern),
    concatenates with CLS token internally, then passes through an MLP head.
    """

    def __init__(
        self,
        state_dim: int,
        cls_dim: int,
        state_hidden_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        # State encoder: 2-layer MLP with LayerNorm (mirrors TransformerQNetwork.state_encoder)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim),
            nn.GELU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.LayerNorm(state_hidden_dim),
        )
        # MLP head over [state_enc || cls_token]
        d = state_hidden_dim + cls_dim
        layers: list[nn.Module] = [nn.GELU()]
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_flat: Tensor, cls_token: Tensor) -> Tensor:
        """
        state_flat : (B, state_dim)
        cls_token  : (B, cls_dim)
        Returns    : (B,)
        """
        state_enc = self.state_encoder(state_flat)
        obs_feat = torch.cat([state_enc, cls_token], dim=-1)
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
    """Repeat obs n times along the batch dim: (B, ...) → (n*B, ...).
    Non-tensor values (e.g. floats/bools injected by transition_to_batch) are passed through."""
    return {
        k: v.repeat(n, *((1,) * (v.dim() - 1))) if isinstance(v, torch.Tensor) else v
        for k, v in obs.items()
    }


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
    from_pretrained(...)            ensures config is loaded as PARLDiffusionConfig
    """

    name = "parl_diffusion"
    config_class = PARLDiffusionConfig

    def __init__(self, config: PARLDiffusionConfig) -> None:
        super().__init__(config)

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

        pcfg = self.config
        if pcfg.q_type == "mlp":
            enc = self.diffusion.rgb_encoder
            patches_per_image = (enc.IMG_SIZE // enc.backbone.patch_size) ** 2
            num_patch = patches_per_image * self._n_obs_steps * self._n_cams
            self.critic_q = MLPQNetwork(
                num_patch=num_patch,
                encoder_token_dim=encoder_token_dim,
                state_dim=state_dim,
                action_dim=self._action_dim,
                n_action_steps=self._n_act,
                spatial_emb=pcfg.q_hidden_dim,
                mlp_dims=pcfg.q_mlp_dims,
                dropout=pcfg.q_dropout,
            )
        else:
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

            total_cls_dim = cls_dim * self._n_obs_steps * self._n_cams
            self.critic_v = VNetwork(state_dim, total_cls_dim, pcfg.q_hidden_dim, pcfg.v_hidden_dims)

        # Frozen pretrained UNet for high-noise steps (t >= num_ft_train_steps).
        # Snapshotted from the base checkpoint before any fine-tuning begins.
        # Included in state_dict() so it is pushed to and loaded by rtc_server, making
        # inference self-contained — no caller needs to know about PARLBCAlgorithm.
        self.pretrained_unet = copy.deepcopy(self.diffusion.unet)
        self.pretrained_unet.requires_grad_(False)
        self.pretrained_unet.eval()

        # Cutoff in DDPM step convention: steps t < num_ft_train_steps use diffusion.unet
        # (fine-tuned); steps t >= num_ft_train_steps use pretrained_unet. 0 = all steps
        # fine-tuned (no split). Stored as a buffer so it travels with state_dict().
        self.register_buffer("_num_ft_train_steps", torch.tensor(0, dtype=torch.long))

        # Set to True after first weight hot-swap from ft_learner.
        # Until then, predict_action_chunk falls back to plain diffusion sampling.
        self._q_initialized: bool = False

    @property
    def num_ft_train_steps(self) -> int:
        return int(self._num_ft_train_steps.item())

    @num_ft_train_steps.setter
    def num_ft_train_steps(self, value: int) -> None:
        self._num_ft_train_steps.fill_(value)

    def train(self, mode: bool = True) -> "PARLDiffusionPolicy":
        """Keep critic_target_q and pretrained_unet permanently in eval mode."""
        super().train(mode)
        if hasattr(self, "critic_target_q"):
            self.critic_target_q.eval()
        self.pretrained_unet.eval()
        return self

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Ensure config is loaded as PARLDiffusionConfig even when the checkpoint was saved
        as a plain DiffusionPolicy (type: diffusion). PARL-specific fields (inference_only,
        n_samples, etc.) use their dataclass defaults unless overridden via --policy.XX.
        strict=False (the default) handles the missing critic keys from a base checkpoint.

        Re-snapshots pretrained_unet from diffusion.unet only when loading from a base
        DiffusionPolicy checkpoint (no pretrained_unet.* keys present). The deepcopy in
        __init__ runs before checkpoint weights are loaded (diffusion.unet is still random
        at that point), so the re-snapshot is needed to capture the correct pretrained weights.

        When resuming from a fine-tuned PARL checkpoint, pretrained_unet.* keys are present
        and already loaded correctly by super().from_pretrained(), so no re-snapshot occurs.
        """
        kwargs.setdefault("config_cls", PARLDiffusionConfig)
        is_parl_ckpt = cls._checkpoint_has_pretrained_unet(pretrained_name_or_path)
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        if not is_parl_ckpt:
            policy.pretrained_unet = copy.deepcopy(policy.diffusion.unet)
            policy.pretrained_unet.requires_grad_(False)
            policy.pretrained_unet.eval()
        return policy

    @classmethod
    def _checkpoint_has_pretrained_unet(cls, pretrained_name_or_path) -> bool:
        """Return True if the safetensors checkpoint contains pretrained_unet.* keys.

        Reads only the safetensors header (no tensor data loaded). Returns False for
        HuggingFace Hub paths (treated as base checkpoints) or on any I/O error.
        """
        import os
        from safetensors import safe_open
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

        model_id = str(pretrained_name_or_path)
        if not os.path.isdir(model_id):
            return False  # Hub path: assume base checkpoint
        model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
        try:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                return any(k.startswith("pretrained_unet.") for k in f.keys())
        except Exception:
            return False

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
        # Rebuild batch from queues: OBS_IMAGES is already the stacked (B, s, n_cams, C, H, W)
        # tensor — individual camera keys (e.g. "observation.images.top") are no longer present.
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        if not self._q_initialized or not self.config.use_q_selection:
            return self.diffusion.generate_actions(batch, noise=noise, full_length=full_length, action_cond=action_cond, noise_injection_std=self.config.noise_injection_std)
        # Do NOT call _stack_images here: OBS_IMAGES is already stacked from the queues above.
        return self._sample_and_select(batch, self.config.n_samples, full_length, action_cond=action_cond)

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

        assert "adv_cond" not in batch, "_sample_and_select does not support advantage conditioning"
        encoder_tokens_n = encoder_tokens.repeat(n_samples, *((1,) * (encoder_tokens.dim() - 1)))
        state_flat_n = state_flat.repeat(n_samples, 1)
        global_cond_n = {"encoder_tokens": encoder_tokens_n, "state": state_flat_n}
        action_cond_n = (
            action_cond.repeat(n_samples, *((1,) * (action_cond.dim() - 1)))
            if action_cond is not None else None
        )
        pt_unet = self.pretrained_unet if self.num_ft_train_steps > 0 else None
        sampled = dm.conditional_sample(
            n_samples * B, global_cond=global_cond_n, action_cond=action_cond_n,
            noise_injection_std=self.config.noise_injection_std,
            pretrained_unet=pt_unet,
            num_ft_train_steps=self.num_ft_train_steps,
        ).view(n_samples, B, self._horizon, self._action_dim)

        q_scores = self._q_score_samples(encoder_tokens, state_flat, sampled)
        best, _ = self._select_best(sampled, q_scores)

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

    def _select_best(self, sampled: Tensor, q_scores: Tensor) -> tuple[Tensor, Tensor]:
        """
        sampled  : (n_samples, B, horizon, Da)
        q_scores : (n_samples, B)
        returns  : (B, horizon, Da), best_idx (B,)
        """
        n, B = sampled.shape[:2]
        if self.config.distil_temp > 0:
            probs = F.softmax(q_scores / self.config.distil_temp, dim=0).T  # (B, n)
            best_idx = Categorical(probs=probs).sample()                       # (B,)
        else:
            best_idx = q_scores.argmax(dim=0)                                  # (B,)

        best = sampled.gather(
            0,
            best_idx[None, :, None, None].expand(1, B, self._horizon, self._action_dim),
        ).squeeze(0)
        return best, best_idx

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
        pcfg = self.config
        obs = _stack_images(batch, self)

        with torch.no_grad():
            encoder_tokens, state_flat, cls_tokens_flat = self.encode_obs(obs)

        action = batch[ACTION][:, :self._n_act]                    # (B, n_act, Da)

        # Gamma-discounted H-step return: G = Σ_{t=0}^{H-1} γ^t * r_t  →  (B,)
        reward_seq = batch[pcfg.reward_key].float().squeeze(-1)    # (B, H, 1) → (B, H)
        B = reward_seq.shape[0]
        gamma_weights = pcfg.gamma ** torch.arange(
            self._horizon, device=reward_seq.device, dtype=reward_seq.dtype
        )                                                           # (H,)

        # Mask padded steps; real_h = number of valid reward timesteps per sample
        reward_is_pad = batch.get("reward_is_pad")                 # (B, H) bool or None
        valid_mask = (~reward_is_pad.to(reward_seq.device) if reward_is_pad is not None
                      else torch.ones(B, self._horizon, device=reward_seq.device, dtype=torch.bool))
        real_h = valid_mask.sum(dim=-1)                            # (B,)
        reward = (reward_seq * gamma_weights * valid_mask).sum(dim=-1)  # (B,)
        assert reward.shape == (B, )

        # terminated at step H: (B, 1, 1) → (B,)
        terminated_raw = batch.get(pcfg.terminated_key)
        terminated = (
            terminated_raw.float().squeeze(-1).squeeze(-1)
            if terminated_raw is not None
            else torch.zeros(B, device=reward.device, dtype=reward.dtype)
        )
        assert terminated.shape == (B, )

        # V loss: expectile regression against frozen target Q
        with torch.no_grad():
            q_ref = self.critic_target_q(encoder_tokens, state_flat, action)  # (B,)
        v = self.critic_v(state_flat, cls_tokens_flat)                         # (B,)
        assert q_ref.shape == v.shape
        diffs = q_ref - v
        weights = torch.where(diffs > 0, pcfg.expectile, 1.0 - pcfg.expectile)
        v_loss = (weights * diffs.pow(2)).mean()

        # Q loss: H-step TD backup — G + γ^H * (1 - terminated) * V(s_H)
        next_obs = self._build_next_obs_batch(batch)
        with torch.no_grad():
            _, next_state_flat, next_cls_flat = self.encode_obs(next_obs)
            next_v = self.critic_v(next_state_flat, next_cls_flat)             # (B,)
            assert next_v.shape == (B, )
        gamma_H = pcfg.gamma ** real_h                             # (B,)
        q_target = (reward + gamma_H * (1.0 - terminated) * next_v).clamp(
            min=pcfg.q_target_clip_min, max=pcfg.q_target_clip_max
        )

        q = self.critic_q(encoder_tokens, state_flat, action)                  # (B,)
        assert q.shape == (B, )
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
        with a delta of n_action_steps / fps.
        """
        pcfg = self.config
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

    def actor_loss(
        self,
        batch: dict,
        ft_diffusion: nn.Module,
        pretrained_unet=None,
        num_ft_train_steps: int = 0,
        actor_alg: str = "bc",
        dpo_beta: float = 1.0,
        dpo_best_k: int = 1,
        dpo_worst_k: int = 1,
        dpo_scale: float = 1.0,
        bc_scale: float = 0.0,
    ) -> tuple[Tensor, dict]:
        """
        1. Sample n_samples full-horizon actions from ft_diffusion (frozen reference, no grad).
        2. Q-score on n_action_steps portion using current policy's obs encoding (no grad).
        3. Select best per state.
        4. Actor loss toward best action (gradients through self.diffusion):
             'bc'    — standard diffusion/flow BC loss (compute_loss).
             'chain' — flow-only; reuses the sampling chain's initial noise (compute_chain_loss).

        Gradients flow through unet AND rgb_encoder (encoder is trained by the actor).
        ft_diffusion is the frozen reference network; only self.diffusion is updated.
        """
        pcfg = self.config
        n = pcfg.n_samples
        obs = _stack_images(batch, self)
        B = batch[ACTION].shape[0]
        device = next(self.parameters()).device

        assert "adv_cond" not in batch, "actor_loss does not support advantage conditioning"

        # Sample from ft_diffusion and Q-score (all no grad).
        # Reuse encoder_tokens from encode_obs for both sampling and Q-scoring to avoid
        # running the rgb_encoder twice and to keep conditioning consistent.
        with torch.no_grad():
            encoder_tokens, state_flat, _ = self.encode_obs(obs)
            encoder_tokens_n = encoder_tokens.repeat(n, *((1,) * (encoder_tokens.dim() - 1)))
            state_flat_n = state_flat.repeat(n, 1)
            global_cond_n = {"encoder_tokens": encoder_tokens_n, "state": state_flat_n}

            sample_out = ft_diffusion.conditional_sample(   # (n*B, H, Da) [+ chain]
                n * B, global_cond=global_cond_n,
                pretrained_unet=pretrained_unet,
                num_ft_train_steps=num_ft_train_steps,
                noise_injection_std=pcfg.noise_injection_std,
                return_chain=(actor_alg == "chain"),
            )
            if actor_alg == "chain":
                sampled_flat, chains_flat = sample_out
                # chains_flat: list of (n*B, H, Da); reshape each to (n, B, H, Da)
                chains_n = [c.view(n, B, self._horizon, self._action_dim) for c in chains_flat]
            else:
                sampled_flat = sample_out

            sampled = sampled_flat.view(n, B, self._horizon, self._action_dim)
            q_scores = self._q_score_samples(encoder_tokens, state_flat, sampled)  # (n, B)
            best_action, best_idx = self._select_best(sampled, q_scores)           # (B, H, Da)

        # Actor loss — grad flows through self.diffusion → rgb_encoder.
        # Restrict to t < num_ft_train_steps so self.unet is only trained on the steps
        # it will actually handle at inference (pretrained_unet covers the rest).
        bc_batch = dict(obs)
        bc_batch[ACTION] = best_action
        bc_batch["action_is_pad"] = torch.zeros(
            B, self._horizon, dtype=torch.bool, device=device
        )
        max_ts = num_ft_train_steps if num_ft_train_steps > 0 else None

        if actor_alg == "dpo":
            return self._actor_loss_dpo_body(
                ft_diffusion, sampled, q_scores, encoder_tokens, state_flat,
                B, device, num_ft_train_steps, dpo_beta, dpo_best_k, dpo_worst_k,
                dpo_scale, bc_scale,
            )

        if actor_alg == "chain":
            # Select the chain corresponding to the best sample for each batch element.
            expand = (1, B, self._horizon, self._action_dim)
            idx = best_idx[None, :, None, None].expand(*expand)
            best_chain = [c.gather(0, idx).squeeze(0) for c in chains_n]  # list of (B, H, Da)
            actor_loss = self.diffusion.compute_chain_loss(bc_batch, best_chain, max_timestep=max_ts)
        else:
            actor_loss = self.diffusion.compute_loss(bc_batch, max_timestep=max_ts)

        if pcfg.distil_temp > 0:
            probs = F.softmax(q_scores / pcfg.distil_temp, dim=0)  # (n, B)
            ent = -(probs * probs.clamp(min=1e-8).log()).sum(0)
            q_ent_global = (ent / math.log(max(n, 2))).mean().item()
        else:
            q_ent_global = 0.0

        stats = {
            "actor_loss": actor_loss.item(),
            "q_best_mean": q_scores.max(dim=0).values.mean().item(),
            "q_spread_mean": (
                q_scores.max(dim=0).values - q_scores.min(dim=0).values
            ).mean().item(),
            "q_ent_global": q_ent_global,
        }
        return actor_loss, stats

    def _actor_loss_dpo_body(
        self,
        ft_diffusion: nn.Module,
        sampled: Tensor,
        q_scores: Tensor,
        encoder_tokens: Tensor,
        state_flat: Tensor,
        B: int,
        device,
        num_ft_train_steps: int,
        dpo_beta: float,
        dpo_best_k: int,
        dpo_worst_k: int,
        dpo_scale: float = 1.0,
        bc_scale: float = 0.0,
    ) -> tuple[Tensor, dict]:
        """
        DPO actor loss using the flow-matching denoising MSE as the log-prob surrogate.

        Among the n_samples actions already sampled and Q-scored:
          - winner a_w: random draw from top dpo_best_k  by Q
          - loser  a_l: random draw from bottom dpo_worst_k by Q

        The same noise ε and timestep t are applied to both, giving:
          loss_w/l = ||v_θ(noisy_w/l, t, s) - (a_w/l - ε)||²  (per-sample, act_steps only)

        DPO implicit reward  ≈  log π_ft(a) - log π_ref(a)  =  loss_ref - loss_ft
        L = -E[ log σ( β · (reward_w - reward_l) ) ]
        """
        # ── select winner and loser from Q-ranked samples ──────────────────
        _, top_idx = q_scores.topk(dpo_best_k,  dim=0, largest=True)   # (best_k,  B)
        _, bot_idx = q_scores.topk(dpo_worst_k, dim=0, largest=False)  # (worst_k, B)

        b_range = torch.arange(B, device=device)
        best_idx = top_idx[torch.randint(dpo_best_k,  (B,), device=device), b_range]  # (B,)
        worst_idx = bot_idx[torch.randint(dpo_worst_k, (B,), device=device), b_range]  # (B,)

        gather = lambda idx: idx[None, :, None, None].expand(1, B, self._horizon, self._action_dim)
        a_w = sampled.gather(0, gather(best_idx)).squeeze(0)   # (B, H, Da)
        a_l = sampled.gather(0, gather(worst_idx)).squeeze(0)  # (B, H, Da)

        # ── flow-matching noise + timestep (same for winner and loser) ─────
        tau_min = (
            1.0 - num_ft_train_steps / self.diffusion.num_inference_steps
            if num_ft_train_steps > 0 else 0.0
        )
        t = tau_min + torch.rand(B, device=device) * (1.0 - tau_min)       # (B,)
        noise = torch.randn(B, self._horizon, self._action_dim, device=device)

        t3 = t[:, None, None]                                               # (B, 1, 1)
        noisy_w = (1 - t3) * noise + t3 * a_w                              # (B, H, Da)
        noisy_l = (1 - t3) * noise + t3 * a_l
        vel_w = a_w - noise                                                  # (B, H, Da)
        vel_l = a_l - noise

        # ── train-time RTC: inpaint early steps (mirrors FlowModel.compute_loss) ─
        timesteps_bh = t[:, None].expand(B, self._horizon).clone()          # (B, H)
        rtc_loss_mask = None  # (B, H) True = inpainted step, excluded from loss
        if False:
        #if self.diffusion.config.rtc_type == "train_time":
            d = torch.randint(0, self.diffusion.config.rtc_delay, (B,), device=device)
            h_idx = torch.arange(self._horizon, device=device).unsqueeze(0)
            rtc_mask = h_idx < d.unsqueeze(1)                               # (B, H)
            timesteps_bh = timesteps_bh.masked_fill(rtc_mask, 1.0)
            expand3 = rtc_mask.unsqueeze(-1).expand_as(noisy_w)
            noisy_w = torch.where(expand3, a_w, noisy_w)
            noisy_l = torch.where(expand3, a_l, noisy_l)
            rtc_loss_mask = rtc_mask

        # ── conditioning: reuse precomputed encoder_tokens / state_flat ─────
        # rgb_encoder is frozen (freeze_image_encoder=True) and deleted from ft_diffusion;
        # both the current unet and the reference unet use the same shared obs encoding.
        global_cond = {"encoder_tokens": encoder_tokens, "state": state_flat}

        # current policy predictions — gradients flow through self.diffusion.unet
        pred_w_ft = self.diffusion.unet(noisy_w, timesteps_bh, global_cond=global_cond)
        pred_l_ft = self.diffusion.unet(noisy_l, timesteps_bh, global_cond=global_cond)

        with torch.no_grad():
            pred_w_ref = ft_diffusion.unet(noisy_w, timesteps_bh, global_cond=global_cond)
            pred_l_ref = ft_diffusion.unet(noisy_l, timesteps_bh, global_cond=global_cond)

        # ── per-sample MSE over act_steps (mean over time and action dims) ─
        act = self._n_act

        def _per_sample_mse(pred: Tensor, target: Tensor) -> Tensor:
            sq = (pred[:, :act] - target[:, :act]).pow(2)                   # (B, act, Da)
            if rtc_loss_mask is None:
                return sq.mean((-1, -2))                                     # (B,)
            unmasked = ~rtc_loss_mask[:, :act].unsqueeze(-1).expand_as(sq)  # (B, act, Da)
            count = unmasked.float().sum((-1, -2)).clamp(min=1)             # (B,)
            return (sq * unmasked).sum((-1, -2)) / count                    # (B,)

        loss_w_ft  = _per_sample_mse(pred_w_ft,  vel_w)
        loss_l_ft  = _per_sample_mse(pred_l_ft,  vel_l)
        loss_w_ref = _per_sample_mse(pred_w_ref, vel_w)
        loss_l_ref = _per_sample_mse(pred_l_ref, vel_l)

        # implicit log-ratio:  loss_ref - loss_ft  ≈  log π_ft - log π_ref
        implicit_reward_w = loss_w_ref - loss_w_ft
        implicit_reward_l = loss_l_ref - loss_l_ft

        dpo_loss = -F.logsigmoid(dpo_beta * (implicit_reward_w - implicit_reward_l)).mean()
        bc_loss = loss_w_ft.mean()
        actor_loss = dpo_scale * dpo_loss + bc_scale * bc_loss

        stats = {
            "actor_loss":        actor_loss.item(),
            "dpo_loss":          dpo_loss.item(),
            "bc_loss":           bc_loss.item(),
            "dpo_impl_reward_w": implicit_reward_w.mean().item(),
            "dpo_impl_reward_l": implicit_reward_l.mean().item(),
            "dpo_margin":        (implicit_reward_w - implicit_reward_l).mean().item(),
            "q_best_mean":       q_scores.max(dim=0).values.mean().item(),
            "q_spread_mean":     (q_scores.max(dim=0).values - q_scores.min(dim=0).values).mean().item(),
            "q_ent_global":      0.0,
        }
        return actor_loss, stats

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target_q(self) -> None:
        """EMA update: target_q ← τ * q + (1 - τ) * target_q."""
        if not hasattr(self, "critic_target_q"):
            raise RuntimeError("update_target_q called on inference_only PARLDiffusionPolicy")
        tau = self.config.tau
        for p, tp in zip(self.critic_q.parameters(), self.critic_target_q.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
