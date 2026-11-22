"""AGLA: Adaptive Global-Local Attention module (core contribution of the paper).

Paper description (Section 2.3, paraphrased because the equations in the .docx
are embedded as OMML objects and are not recoverable as plain text):

    global branch  : average-pool feature map with stride ``s`` (s = 4),
                     apply standard multi-head self-attention on the pooled
                     tokens, then bilinearly upsample back to the original
                     resolution.
    local branch   : keep the original resolution, split the feature map into
                     non-overlapping ``w x w`` windows (w = 8), apply window
                     self-attention inside every window.
    fusion gate    : a lightweight 3x3 conv predicts two spatially-varying
                     logits; softmax over the channel dimension yields weights
                     that sum to one; branch outputs are combined elementwise.

The module keeps the token count unchanged, so it can replace the self-
attention layer of a ViT / VGGT encoder block without touching the decoder.

Expected tensor layout
----------------------
``x`` has shape ``(B, N, C)`` with ``N = num_views * H * W``.  The caller also
passes ``grid_hw=(H, W)`` and ``num_views`` so that the module can restore the
spatial/view structure.  ``scope`` controls which tokens the global branch
attends over:

* ``"intra"``  -- attention is computed independently inside each view
  (VGGT's frame-wise / intra-frame attention block);
* ``"cross"``  -- attention is computed jointly over all views
  (VGGT's global / cross-view attention block).

The local branch is always computed per view.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FusionMode = Literal["gated", "concat", "global_only", "local_only"]
AttentionScope = Literal["intra", "cross"]


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention over a flat token sequence.

    Kept separate from the local variant so that pretrained VGGT attention
    weights (``qkv`` / ``proj``) can be copied into the global branch directly.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:
        x: ``(B, N, C)``.
        attn_mask: optional bool mask ``(B, N)`` or ``(B, N, N)``;
            ``True`` means *masked out*.
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            attn = attn.masked_fill(mask, float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class WindowSelfAttention(nn.Module):
    """Non-overlapping window self-attention with a learnable relative bias.

    Input tokens are a single feature map ``(B, H*W, C)``.  When ``H``/``W``
    are not multiples of ``window_size`` the map is padded on the right/bottom
    and cropped after attention, so any resolution works.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        n = window_size
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(n, dtype=torch.long),
                torch.arange(n, dtype=torch.long),
                indexing="ij",
            )
        ).flatten(1)  # (2, n*n)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, n*n, n*n)
        rel = rel.permute(1, 2, 0).contiguous()  # (n*n, n*n, 2)
        rel[..., 0] += n - 1
        rel[..., 1] += n - 1
        rel[..., 0] *= 2 * n - 1
        relative_index = rel.sum(dim=-1)  # (n*n, n*n)
        self.register_buffer("relative_index", relative_index, persistent=False)
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * n - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Args:
        x: ``(B, H*W, C)`` tokens of **one** view.
        h, w: spatial grid size of that view's token map.
        """
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f"token count {n} does not match grid {h}x{w}")
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        feat = x.transpose(1, 2).reshape(b, c, h, w)
        if pad_h or pad_w:
            feat = F.pad(feat, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        nh, nw = hp // ws, wp // ws

        # (B, nh, ws, nw, ws, C) -> (B*nh*nw, ws*ws, C)
        feat = feat.reshape(b, c, nh, ws, nw, ws)
        feat = feat.permute(0, 2, 4, 3, 5, 1).reshape(b * nh * nw, ws * ws, c)

        qkv = self.qkv(feat).reshape(b * nh * nw, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_bias_table[self.relative_index]  # (ws^2, ws^2, H)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, ws^2, ws^2)
        attn = attn + bias
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b * nh * nw, ws * ws, c)
        out = self.proj_drop(self.proj(out))

        out = out.reshape(b, nh, nw, ws, ws, c)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, c, hp, wp)
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out.reshape(b, n, c)


class AGLA(nn.Module):
    """Adaptive Global-Local Attention (drop-in replacement for MHSA).

    Args:
        dim: token/channel dimension.
        num_heads: attention heads used by both branches.
        global_downsample: stride ``s`` of the global branch's average pooling.
        window_size: window side ``w`` of the local branch.
        mode: ``gated`` (full method), or one of the ablation variants
            ``concat`` / ``global_only`` / ``local_only``.
        attn_drop, proj_drop, qkv_bias: standard attention hyper-parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        global_downsample: int = 4,
        window_size: int = 8,
        mode: FusionMode = "gated",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("gated", "concat", "global_only", "local_only"):
            raise ValueError(f"unknown fusion mode: {mode}")
        if global_downsample < 1:
            raise ValueError("global_downsample must be >= 1")
        self.dim = dim
        self.num_heads = num_heads
        self.global_downsample = global_downsample
        self.window_size = window_size
        self.mode: FusionMode = mode

        self.norm_global = nn.LayerNorm(dim)
        self.norm_local = nn.LayerNorm(dim)
        self.global_attn = MultiHeadSelfAttention(
            dim, num_heads, attn_drop, proj_drop, qkv_bias
        )
        self.local_attn = WindowSelfAttention(
            dim, num_heads, window_size, attn_drop, proj_drop, qkv_bias
        )

        if mode == "concat":
            self.fuse = nn.Linear(2 * dim, dim)
        if mode == "gated":
            # Two-channel content embedding -> spatially varying fusion weights.
            self.gate_conv = nn.Conv2d(dim, 2, kernel_size=3, padding=1, bias=True)
            # Start from an exact 0.5 / 0.5 gate: small weights, zero bias.
            nn.init.normal_(self.gate_conv.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.gate_conv.bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.global_attn.qkv.weight)
        nn.init.xavier_uniform_(self.global_attn.proj.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        # Local branch starts from small random values (paper Sec. 2.3.3).
        nn.init.normal_(self.local_attn.qkv.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.local_attn.proj.weight, mean=0.0, std=0.01)
        for module in (self.global_attn, self.local_attn):
            for name, param in module.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
        if hasattr(self, "fuse"):
            nn.init.xavier_uniform_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)

    def load_vggt_attention_state(
        self, state_dict: dict, prefix: str = "", strict: bool = False
    ) -> list[str]:
        """Copy pretrained VGGT self-attention weights into the global branch.

        ``state_dict`` keys are expected to look like ``<prefix>qkv.weight`` and
        ``<prefix>proj.weight`` (possibly with ``.`` separators).  Missing keys
        are simply skipped when ``strict=False`` (default) so that new local /
        gate parameters keep their paper-prescribed initialization.

        Returns the list of keys that were successfully copied.
        """
        copied = []
        mapping = {
            f"{prefix}qkv.weight": self.global_attn.qkv.weight,
            f"{prefix}qkv.bias": self.global_attn.qkv.bias,
            f"{prefix}proj.weight": self.global_attn.proj.weight,
            f"{prefix}proj.bias": self.global_attn.proj.bias,
        }
        for key, target in mapping.items():
            src = state_dict.get(key)
            if src is None:
                if strict:
                    raise KeyError(f"pretrained key not found: {key}")
                continue
            if tuple(src.shape) != tuple(target.shape):
                raise ValueError(
                    f"shape mismatch for {key}: {tuple(src.shape)} vs {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(src)
            copied.append(key)
        return copied

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
        num_views: int,
        scope: AttentionScope = "cross",
        return_gate: bool = False,
    ):
        """Forward pass.

        Args:
            x: ``(B, V*H*W, C)`` tokens.
            grid_hw: ``(H, W)`` token grid of a single view.
            num_views: ``V``.
            scope: ``"intra"`` (per-view global attention) or ``"cross"``
                (global attention over all views at the pooled resolution).
            return_gate: also return the ``(B, V, 2, H, W)`` fusion weights.

        Returns:
            ``out`` or ``(out, gate)`` with the same token shape as ``x``.
        """
        if scope not in ("intra", "cross"):
            raise ValueError(f"unknown attention scope: {scope}")
        b, n, c = x.shape
        h, w = grid_hw
        v = num_views
        if n != v * h * w:
            raise ValueError(
                f"token count {n} != num_views*H*W = {v}*{h}*{w}; "
                "pass grid_hw / num_views consistent with x"
            )
        if c != self.dim:
            raise ValueError(f"channel mismatch: {c} != {self.dim}")
        s = self.global_downsample

        # ---------------- global branch ----------------
        x5 = x.reshape(b, v, h, w, c)
        feat = x5.reshape(b * v, h, w, c).permute(0, 3, 1, 2)  # (BV, C, H, W)
        pooled = F.avg_pool2d(feat, kernel_size=s, stride=s) if s > 1 else feat
        hg, wg = pooled.shape[-2:]
        pooled_tokens = pooled.permute(0, 2, 3, 1).reshape(b * v, hg * wg, c)

        if scope == "cross":
            # Joint attention over all views, on the downsampled token map.
            cross_tokens = pooled_tokens.reshape(b, v, hg * wg, c).reshape(
                b, v * hg * wg, c
            )
            attn = self.global_attn(self.norm_global(cross_tokens))
            attn = attn.reshape(b, v, hg, wg, c).reshape(b * v, hg, wg, c)
        else:
            attn = self.global_attn(self.norm_global(pooled_tokens))
            attn = attn.reshape(b * v, hg, wg, c)

        attn_map = attn.permute(0, 3, 1, 2)  # (BV, C, hg, wg)
        up = F.interpolate(
            attn_map, size=(h, w), mode="bilinear", align_corners=False
        )
        global_out = up.permute(0, 2, 3, 1).reshape(b, v, h, w, c)

        # ---------------- local branch ----------------
        local_tokens = x5.reshape(b * v, h * w, c)
        local_out = self.local_attn(self.norm_local(local_tokens), h, w)
        local_out = local_out.reshape(b, v, h, w, c)

        # ---------------- fusion ----------------
        gate: Optional[torch.Tensor] = None
        if self.mode == "global_only":
            fused = global_out
        elif self.mode == "local_only":
            fused = local_out
        elif self.mode == "concat":
            fused = self.fuse(torch.cat([global_out, local_out], dim=-1))
        else:  # gated
            logits = self.gate_conv(x5.reshape(b * v, c, h, w))
            gate = logits.reshape(b, v, 2, h, w).softmax(dim=2)
            gw = gate.permute(0, 1, 3, 4, 2)  # (B, V, H, W, 2)
            fused = gw[..., 0:1] * global_out + gw[..., 1:2] * local_out

        out = self.proj_drop(self.proj(fused)).reshape(b, n, c)
        if return_gate:
            return out, gate
        return out


class AGLABlock(nn.Module):
    """Pre-norm Transformer block with AGLA in place of self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        global_downsample: int = 4,
        window_size: int = 8,
        mode: FusionMode = "gated",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AGLA(
            dim=dim,
            num_heads=num_heads,
            global_downsample=global_downsample,
            window_size=window_size,
            mode=mode,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_drop),
        )
        # Stochastic depth is implemented as a simple scaling of residuals.
        self.drop_path_rate = drop_path

    def _residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.drop_path_rate > 0 and self.training:
            keep = 1.0 - self.drop_path_rate
            mask = x.new_empty(x.shape[:1] + (1,) * (x.dim() - 1)).bernoulli_(keep)
            y = y * mask / keep
        return x + y

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
        num_views: int,
        scope: AttentionScope = "cross",
        return_gate: bool = False,
    ):
        y, gate = self.attn(
            self.norm1(x), grid_hw, num_views, scope=scope, return_gate=True
        )
        x = self._residual(x, y)
        x = self._residual(x, self.mlp(self.norm2(x)))
        if return_gate:
            return x, gate
        return x
