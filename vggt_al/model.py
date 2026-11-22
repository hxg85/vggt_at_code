"""Lightweight VGGT-style encoder/decoder with AGLA replacing self-attention.

This is a research-friendly re-implementation of the architecture described in
Section 2.1 / 2.3.3 of the paper.  It follows the VGGT design choices that
matter for the method -- convolutional patch stem, alternating frame-wise and
cross-view attention, and depth / camera / point-cloud heads -- while keeping
the parameter count small enough to run on a single GPU.  Pretrained VGGT
attention weights can be loaded into the global branch through
``VGGTAL.init_from_vggt``; the local branch and fusion gate keep their
paper-prescribed initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AGLABlock, AttentionScope, FusionMode


@dataclass
class VGGTALConfig:
    """All model hyper-parameters.  Defaults follow Section 3.1 of the paper."""

    img_size: Tuple[int, int] = (384, 512)  # (H, W), paper's 512x384 images
    patch_size: int = 16  # token grid = 24 x 32 at default resolution
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_views: int = 6
    global_downsample: int = 4
    window_size: int = 8
    fusion_mode: FusionMode = "gated"
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.0
    depth_head_dim: int = 128
    camera_hidden_dim: int = 256
    min_depth: float = 0.05
    max_depth: float = 200.0
    # Set to True to insert a cross-view ("global") attention block first;
    # VGGT alternates frame-wise and global attention blocks.
    cross_first: bool = False
    block_scopes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        h, w = self.img_size
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"img_size {(h, w)} must be divisible by patch_size {self.patch_size}"
            )
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.fusion_mode not in ("gated", "concat", "global_only", "local_only"):
            raise ValueError(f"unknown fusion mode: {self.fusion_mode}")
        if self.depth < 1 or self.num_views < 1:
            raise ValueError("depth and num_views must be positive")

    def scopes(self) -> List[AttentionScope]:
        """Per-block attention scope, alternating intra-view and cross-view."""
        if self.block_scopes:
            if len(self.block_scopes) != self.depth:
                raise ValueError("block_scopes length must equal depth")
            return list(self.block_scopes)  # type: ignore[return-value]
        first: AttentionScope = "cross" if self.cross_first else "intra"
        second: AttentionScope = "intra" if self.cross_first else "cross"
        return [first if i % 2 == 0 else second for i in range(self.depth)]


class PatchEmbed(nn.Module):
    """Convolutional stem: non-overlapping patches -> token embeddings."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: ``(B*V, 3, H, W)`` -> ``(B*V, C, h, w)``."""
        return self.proj(images)


class DepthHead(nn.Module):
    """Predict per-view depth and confidence from the token feature map."""

    def __init__(self, dim: int, head_dim: int, min_depth: float, max_depth: float) -> None:
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.net = nn.Sequential(
            nn.Conv2d(dim, head_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_dim, head_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_dim // 2, 2, kernel_size=1),
        )

    def forward(
        self, tokens: torch.Tensor, h: int, w: int, out_hw: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokens: ``(B*V, h*w, C)`` -> depth/confidence ``(B*V, 1, H_img, W_img)``."""
        bv, _, c = tokens.shape
        feat = tokens.transpose(1, 2).reshape(bv, c, h, w)
        feat = F.interpolate(feat, size=out_hw, mode="bilinear", align_corners=False)
        pred = self.net(feat)
        raw_depth = F.softplus(pred[:, 0:1]) + self.min_depth
        depth = torch.clamp(raw_depth, max=self.max_depth)
        confidence = torch.sigmoid(pred[:, 1:2])
        return depth, confidence


class CameraHead(nn.Module):
    """Predict camera-to-world matrices (translation + quaternion)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7),
        )

    @staticmethod
    def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
        """Normalized quaternions ``(..., 4)`` in (w, x, y, z) -> rotation matrices."""
        q = F.normalize(q, dim=-1, eps=1e-8)
        w, x, y, z = q.unbind(dim=-1)
        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rot = torch.stack(
            [
                1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
                2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
                2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2),
            ],
            dim=-1,
        )
        return rot.reshape(*q.shape[:-1], 3, 3)

    def forward(self, tokens: torch.Tensor, num_views: int) -> torch.Tensor:
        """tokens: ``(B, V*N, C)`` -> cameras ``(B, V, 4, 4)`` (camera-to-world)."""
        b, _, c = tokens.shape
        v = num_views
        per_view = tokens.reshape(b, v, -1, c)
        pooled = torch.cat(
            [per_view.mean(dim=2), per_view.max(dim=2).values], dim=-1
        )  # (B, V, 2C)
        out = self.mlp(pooled)  # (B, V, 7): tx,ty,tz,qw,qx,qy,qz
        trans, quat = out[..., :3], out[..., 3:]
        rot = self.quaternion_to_rotation(quat)
        cameras = torch.eye(4, device=tokens.device, dtype=tokens.dtype)
        cameras = cameras.view(1, 1, 4, 4).expand(b, v, -1, -1).clone()
        cameras[..., :3, :3] = rot
        cameras[..., :3, 3] = trans
        return cameras


def default_intrinsics(height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """Pinhole prior used by the Tanks and Temples benchmark.

    Principal point at image centre, focal length ``0.7 * W``; returned as
    ``(1, 1, 3, 3)`` so it broadcasts over batch and views.
    """
    k = torch.zeros(3, 3, device=device, dtype=dtype)
    k[0, 0] = k[1, 1] = 0.7 * float(width)
    k[0, 2] = 0.5 * float(width)
    k[1, 2] = 0.5 * float(height)
    k[2, 2] = 1.0
    return k.view(1, 1, 3, 3)


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    cameras: torch.Tensor,
) -> torch.Tensor:
    """Lift per-view depth maps to world coordinates.

    Args:
        depth: ``(B, V, 1, H, W)``.
        intrinsics: ``(B, V, 3, 3)`` or broadcastable.
        cameras: ``(B, V, 4, 4)`` camera-to-world transforms.

    Returns:
        ``(B, V, H, W, 3)`` world-space points.
    """
    b, v, _, h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0].reshape(b, v, 1, 1)
    fy = intrinsics[..., 1, 1].reshape(b, v, 1, 1)
    cx = intrinsics[..., 0, 2].reshape(b, v, 1, 1)
    cy = intrinsics[..., 1, 2].reshape(b, v, 1, 1)
    z = depth[:, :, 0]  # (B, V, H, W)
    px = (xs - cx) / fx.clamp_min(1e-6) * z
    py = (ys - cy) / fy.clamp_min(1e-6) * z
    pts_cam = torch.stack([px, py, z], dim=-1).reshape(b * v, h * w, 3)
    rot = cameras[..., :3, :3].reshape(b * v, 3, 3)
    trans = cameras[..., :3, 3].reshape(b * v, 3)
    pts_world = torch.bmm(pts_cam, rot.transpose(1, 2)) + trans[:, None, :]
    return pts_world.reshape(b, v, h, w, 3)


class VGGTAL(nn.Module):
    """VGGT-AL: VGGT-style reconstruction network with AGLA encoder blocks."""

    def __init__(self, config: Optional[VGGTALConfig] = None) -> None:
        super().__init__()
        self.cfg = config or VGGTALConfig()
        self.cfg.validate()
        cfg = self.cfg
        self.token_h = cfg.img_size[0] // cfg.patch_size
        self.token_w = cfg.img_size[1] // cfg.patch_size

        self.patch_embed = PatchEmbed(3, cfg.embed_dim, cfg.patch_size)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, self.token_h, self.token_w, cfg.embed_dim)
        )
        self.view_embed = nn.Parameter(torch.zeros(1, cfg.num_views, 1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        dpr = [cfg.drop_path * i / max(cfg.depth - 1, 1) for i in range(cfg.depth)]
        self.blocks = nn.ModuleList(
            [
                AGLABlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    global_downsample=cfg.global_downsample,
                    window_size=cfg.window_size,
                    mode=cfg.fusion_mode,
                    attn_drop=cfg.attn_drop,
                    proj_drop=cfg.proj_drop,
                    drop_path=dpr[i],
                )
                for i in range(cfg.depth)
            ]
        )
        self.scopes: List[AttentionScope] = cfg.scopes()
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.depth_head = DepthHead(
            cfg.embed_dim, cfg.depth_head_dim, cfg.min_depth, cfg.max_depth
        )
        self.camera_head = CameraHead(cfg.embed_dim, cfg.camera_hidden_dim)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_from_vggt(self, state_dict: dict, verbose: bool = True) -> Tuple[int, int]:
        """Initialize global branches from a pretrained VGGT state dict.

        Matches keys of the form ``blocks.<i>.attn.qkv.weight`` (and optional
        prefixes such as ``model.`` or ``encoder.``) to our
        ``blocks.<i>.attn.global_attn.qkv.weight``.  Also copies norms / MLPs
        when present.  Returns ``(copied, skipped)`` counts.
        """
        copied = 0
        skipped = 0
        for i, block in enumerate(self.blocks):
            candidates = []
            for outer in ("", "model.", "encoder.", "module."):
                candidates.append(f"{outer}blocks.{i}.attn.")
            src_prefix = next((p for p in candidates if any(k.startswith(p) for k in state_dict)), None)
            if src_prefix is None:
                skipped += len(block.attn.global_attn.state_dict())
                continue
            target_state = block.state_dict()
            new_state: Dict[str, torch.Tensor] = {}
            for key, value in target_state.items():
                if key.startswith("attn.local_attn") or "gate_conv" in key or "fuse" in key:
                    continue  # new parameters: keep paper initialization
                src_key = src_prefix + key.replace("attn.global_attn.", "attn.")
                src = state_dict.get(src_key)
                if src is not None and tuple(src.shape) == tuple(value.shape):
                    new_state[key] = src
                    copied += 1
                else:
                    skipped += 1
            if new_state:
                block.load_state_dict(new_state, strict=False)
        if verbose:
            print(f"[VGGT-AL] copied {copied} tensors from VGGT; {skipped} kept/skipped")
        return copied, skipped

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        return_gates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run feed-forward reconstruction.

        Args:
            images: ``(B, V, 3, H, W)`` RGB tensors (values in [0, 1] or
                normalized with the same transform used during training).
            intrinsics: optional ``(B, V, 3, 3)``; when omitted the Tanks and
                Temples pinhole prior is used.
            return_gates: collect per-block fusion weight maps.

        Returns:
            dict with ``depth`` (B,V,1,H,W), ``confidence``, ``cameras``
            (B,V,4,4), ``points`` (B,V,H,W,3), optional ``gates`` and ``tokens``.
        """
        if images.dim() != 5:
            raise ValueError("images must be (B, V, 3, H, W)")
        b, v, _, hi, wi = images.shape
        if v > self.cfg.num_views:
            raise ValueError(
                f"got {v} views but model was built for {self.cfg.num_views}; "
                "set VGGTALConfig.num_views accordingly"
            )
        if intrinsics is None:
            intrinsics = default_intrinsics(hi, wi, images.device, images.dtype).expand(
                b, v, -1, -1
            )
        if intrinsics.shape != (b, v, 3, 3):
            raise ValueError(f"intrinsics must be (B,V,3,3), got {tuple(intrinsics.shape)}")

        flat = images.reshape(b * v, 3, hi, wi)
        feat = self.patch_embed(flat)  # (BV, C, h, w)
        h, w = feat.shape[-2:]
        if (h, w) != (self.token_h, self.token_w):
            raise ValueError(
                f"input resolution {(hi, wi)} gives token grid {(h, w)}, "
                f"expected {(self.token_h, self.token_w)}"
            )
        tokens = feat.flatten(2).transpose(1, 2)  # (BV, N, C)
        tokens = tokens.reshape(b, v, h * w, self.cfg.embed_dim)
        tokens = (
            tokens
            + self.spatial_pos[:, :, :h, :w, :].reshape(1, 1, h * w, self.cfg.embed_dim)
            + self.view_embed[:, :v, :, :].reshape(1, v, 1, self.cfg.embed_dim)
        )
        tokens = tokens.reshape(b, v * h * w, self.cfg.embed_dim)

        gates: List[torch.Tensor] = []
        for scope, block in zip(self.scopes, self.blocks):
            if return_gates:
                tokens, gate = block(
                    tokens, (h, w), v, scope=scope, return_gate=True
                )
                if gate is not None:
                    gates.append(gate)
            else:
                tokens = block(tokens, (h, w), v, scope=scope)

        tokens = self.norm(tokens)
        per_view_tokens = tokens.reshape(b * v, h * w, self.cfg.embed_dim)
        depth, confidence = self.depth_head(per_view_tokens, h, w, (hi, wi))
        depth = depth.reshape(b, v, 1, hi, wi)
        confidence = confidence.reshape(b, v, 1, hi, wi)
        cameras = self.camera_head(tokens, v)
        points = backproject_depth(depth, intrinsics, cameras)

        out: Dict[str, torch.Tensor] = {
            "depth": depth,
            "confidence": confidence,
            "cameras": cameras,
            "points": points,
        }
        if return_gates:
            out["gates"] = torch.stack(gates, dim=1) if gates else torch.empty(0)
            out["tokens"] = tokens
        return out
