"""Block-causal transformer for masked patch reconstruction.

Token layout (flattened): `[(t=0, p=0..N-1), (t=1, p=0..N-1), ...]`
where `N = (H/P) * (W/P)` is the number of spatial patches per frame.

Attention is "block-causal" along the temporal axis: a token at time `t`
may attend to all tokens at times `0..t` (including itself), but not to
future tokens. Within a single time index, attention is bidirectional.

When `enable_timestep=True` the model also accepts a per-sample
diffusion timestep `t: (B,)` and adds a sinusoidal+MLP embedding to all
token features (variant A''). In this mode the input clip is expected
to already contain noised patches and the `mask` argument is typically
omitted.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> (B, T, N, D)
        B, T, C, H, W = x.shape
        x = self.proj(x.reshape(B * T, C, H, W))
        return rearrange(x, "(b t) d h w -> b t (h w) d", b=B, t=T)


class BlockCausalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, S, dh)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BlockCausalAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def block_causal_mask(num_frames: int, num_patches_per_frame: int, device=None) -> torch.Tensor:
    """Bool mask of shape (S, S), True = attend, where S = T * N.

    Token i can attend to token j iff `frame_id(j) <= frame_id(i)`.
    """
    T, N = num_frames, num_patches_per_frame
    fid = torch.arange(T * N, device=device) // N  # (S,)
    return fid[None, :] <= fid[:, None]


class ActionEmbedding(nn.Module):
    """Per-frame continuous action vector -> dim-D embedding via small MLP."""

    def __init__(self, action_dim: int, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, T, A) -> (B, T, D)
        return self.proj(actions)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep -> MLP -> dim-D vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """`t` may be `(B,)` (per-sample) or `(B, T)` (per-frame).

        Returns an embedding broadcastable along token positions: shape
        `(B, 1, 1, D)` for per-sample or `(B, T, 1, D)` for per-frame.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half)
        )
        a = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class BlockCausalVideoTransformer(nn.Module):
    """Patch-based block-causal transformer with masked-token inputs.

    Input: `clip` of shape `(B, T, C, H, W)` and a bool `mask_indicator`
    of shape `(B, T, N)`. After patch embedding, masked patch tokens are
    replaced with a learned `[MASK]` token.

    Output: per-patch pixel predictions of shape `(B, T, N, P*P*C)`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        num_frames: int = 16,
        image_size: int = 128,
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        enable_timestep: bool = False,
        action_dim: int = 0,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        per_side = image_size // patch_size
        self.num_patches_per_frame = per_side * per_side
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.patch_embed = PatchEmbed(in_channels, dim, patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, self.num_patches_per_frame, dim)
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)
        self.enable_timestep = enable_timestep
        if enable_timestep:
            self.time_embed = TimestepEmbedding(dim)
        self.action_dim = int(action_dim)
        if self.action_dim > 0:
            self.action_embed = ActionEmbedding(self.action_dim, dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def _trunk(
        self,
        clip: torch.Tensor,
        mask_indicator: torch.Tensor | None,
        t: torch.Tensor | None,
        actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shared transformer trunk producing post-norm token features
        of shape ``(B, T, N, D)``."""
        B, T, C, H, W = clip.shape
        x = self.patch_embed(clip)  # (B, T, N, D)
        N = x.shape[2]
        if mask_indicator is not None:
            m = mask_indicator.unsqueeze(-1).to(x.dtype)
            x = x * (1.0 - m) + self.mask_token * m
        x = x + self.pos_embed[:, :T, :N]
        if self.enable_timestep:
            assert t is not None, "diffusion-mode model requires t"
            te = self.time_embed(t).to(x.dtype)  # (B, D) or (B, T, D)
            if te.dim() == 2:
                x = x + te[:, None, None, :]
            else:
                x = x + te[:, :, None, :]
        if self.action_dim > 0 and actions is not None:
            ae = self.action_embed(actions).to(x.dtype)  # (B, T, D)
            x = x + ae[:, :, None, :]
        x = rearrange(x, "b t n d -> b (t n) d")
        attn_mask = block_causal_mask(T, N, device=x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)
        return rearrange(x, "b (t n) d -> b t n d", t=T, n=N)

    def forward(
        self,
        clip: torch.Tensor,
        mask_indicator: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        ``mask_indicator``: (B,T,N) bool, True positions get the learned
        mask_token swapped into their patch embedding. Used for variant
        (A) inpainting; pass ``None`` for diffusion (no swap).

        ``t``: (B,) integer timestep tensor. Required when
        ``enable_timestep=True``; adds a learned timestep embedding to
        every token before transformer blocks.
        """
        if self.action_dim > 0 and actions is None:
            raise AssertionError("action-conditioned model requires actions")
        x = self._trunk(clip, mask_indicator, t, actions)
        return self.head(x)

    @torch.no_grad()
    def forward_features(
        self,
        clip: torch.Tensor,
        mask_indicator: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return post-norm transformer features ``(B, T, N, D)``, no head.

        For action-conditioned models, ``actions`` may be ``None`` -- the
        action embedding is simply skipped. This makes downstream probes
        for action prediction non-trivial (no leakage from action input).
        """
        return self._trunk(clip, mask_indicator, t, actions)
