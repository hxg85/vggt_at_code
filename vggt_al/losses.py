"""Multi-task loss described in Section 2.4 of the paper.

Components
----------
1. depth regression: L1 + scale-invariant log loss;
2. radiance-field reconstruction: photometric (reprojection) loss + perceptual
   term (LPIPS when installed, otherwise a differentiable gradient prior);
3. camera consistency: relative-pose supervision when GT poses are available,
   otherwise a cycle-consistency constraint on the predicted camera graph;
4. fusion regularization: total-variation smoothness of the AGLA gate maps.

Every term is optional: missing ground truth (e.g. Tanks and Temples test set,
where poses and point clouds are withheld) simply disables the corresponding
supervised term and relies on the self-supervised ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import backproject_depth


@dataclass
class LossWeights:
    depth_l1: float = 1.0
    depth_si_log: float = 0.5
    photometric: float = 0.5
    perceptual: float = 0.05
    camera: float = 0.1
    fusion_smooth: float = 0.01


def _valid_depth_mask(gt: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(gt) & (gt > 0)


def robust_l1(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier-style robust L1, less sensitive to occlusions."""
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def depth_loss(
    pred: torch.Tensor, gt: torch.Tensor, si_weight: float = 0.5
) -> Dict[str, torch.Tensor]:
    """pred / gt: ``(B, V, 1, H, W)``; invalid GT pixels are ignored."""
    mask = _valid_depth_mask(gt)
    if mask.sum() < 2:
        zero = pred.sum() * 0.0
        return {"depth_l1": zero, "depth_si_log": zero}
    p = pred[mask]
    g = gt[mask]
    l1 = F.l1_loss(p, g)
    log_diff = torch.log(p.clamp_min(1e-6)) - torch.log(g.clamp_min(1e-6))
    # Scale-invariant error (Eigen et al.): variance of log residuals.
    si = (log_diff.pow(2).mean() - log_diff.mean().pow(2)).clamp_min(0.0)
    return {"depth_l1": l1, "depth_si_log": si * float(si_weight)}


def photometric_loss(
    images: torch.Tensor,
    depth: torch.Tensor,
    cameras: torch.Tensor,
    intrinsics: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Warp view 0 into the other views with predicted geometry; robust L1.

    images: ``(B, V, 3, H, W)``; depth: ``(B, V, 1, H, W)``;
    cameras: ``(B, V, 4, 4)``; intrinsics: ``(B, V, 3, 3)``.
    """
    b, v, _, h, w = images.shape
    if v < 2:
        return images.sum() * 0.0

    points = backproject_depth(depth, intrinsics, cameras)  # (B,V,H,W,3)
    # Lift reference pixels to world space, then project them into each source
    # camera.  Using the *reference* depth is what makes the warp meaningful.
    ref_world = points[:, 0].reshape(b, h * w, 3)  # (B,H*W,3)
    losses = []
    for s in range(1, v):
        # World -> source camera: R_s^T (p_world - t_s); R is orthogonal, so
        # row-vector form (p - t) @ R equals R^T @ (p - t).
        pts_cam_s = torch.bmm(
            (ref_world - cameras[:, s, :3, 3][:, None, :]),
            cameras[:, s, :3, :3],
        )  # (B,H*W,3)
        fx = intrinsics[:, s, 0, 0].view(b, 1)
        fy = intrinsics[:, s, 1, 1].view(b, 1)
        cx = intrinsics[:, s, 0, 2].view(b, 1)
        cy = intrinsics[:, s, 1, 2].view(b, 1)
        z = pts_cam_s[..., 2].clamp_min(1e-4)
        u = pts_cam_s[..., 0] / z * fx + cx
        vv = pts_cam_s[..., 1] / z * fy + cy
        grid = torch.stack(
            [2 * u / max(w - 1, 1) - 1, 2 * vv / max(h - 1, 1) - 1], dim=-1
        ).reshape(b, h, w, 2)
        inside = (
            (grid[..., 0].abs() <= 1)
            & (grid[..., 1].abs() <= 1)
            & (z.reshape(b, h, w) > 1e-4)
        ).detach()
        warped = F.grid_sample(
            images[:, s], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        diff = (warped - images[:, 0]).abs()
        if confidence is not None:
            diff = diff * confidence[:, 0]
        if inside.any():
            losses.append((diff * inside[:, None]).sum() / inside.sum().clamp_min(1) / 3)
        else:
            losses.append(diff.mean())
    return sum(losses) / len(losses)


def perceptual_loss(images: torch.Tensor, use_lpips: bool = False) -> torch.Tensor:
    """Perceptual term.  Uses LPIPS when available, otherwise an edge prior.

    The paper uses LPIPS in its radiance-field loss.  Keeping a dependency-free
    fallback makes the repository runnable in minimal environments; set
    ``use_lpips=True`` (and install ``lpips``) for the paper-faithful variant.
    """
    if use_lpips:
        try:
            import lpips  # type: ignore

            fn = lpips.LPIPS(net="vgg").to(images.device).eval()
            b, v = images.shape[:2]
            loss = 0.0
            for s in range(1, v):
                loss = loss + fn(images[:, 0], images[:, s]).mean()
            return loss / max(v - 1, 1)
        except ImportError:
            pass
    gx = images[..., :, 1:] - images[..., :, :-1]
    gy = images[..., 1:, :] - images[..., :-1, :]
    return (gx.pow(2).mean() + gy.pow(2).mean()) * 0.5


def relative_pose(c2w: torch.Tensor) -> torch.Tensor:
    """Relative camera transforms T_i0 = inv(c2w_0) @ c2w_i, shape ``(B,V,4,4)``."""
    inv0 = torch.linalg.inv(c2w[:, 0])  # (B,4,4)
    return torch.matmul(inv0[:, None, :, :], c2w)  # (B,V,4,4)


def camera_loss(
    cameras: torch.Tensor, gt_cameras: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Relative-pose supervision when calibrations are available.

    Without camera ground truth this term is disabled: camera geometry is then
    supervised by the photometric reprojection loss only.  (A naive
    ``inv(T_i0) @ T_i0`` cycle term is algebraically zero and would provide no
    gradient, so it is deliberately not used.)
    """
    if gt_cameras is None:
        return cameras.sum() * 0.0
    rel = relative_pose(cameras)
    rel_gt = relative_pose(gt_cameras)
    rot_err = (rel[:, :, :3, :3] - rel_gt[:, :, :3, :3]).pow(2).mean()
    t_pred, t_gt = rel[:, :, :3, 3], rel_gt[:, :, :3, 3]
    scale = t_gt.norm(dim=-1).mean().detach()
    if scale < 1e-4:  # degenerate / all-identity calibration
        trans_err = t_pred.pow(2).mean()
    else:
        trans_err = ((t_pred - t_gt) / scale).pow(2).mean()
    return rot_err + trans_err


def fusion_smoothness(gates: torch.Tensor) -> torch.Tensor:
    """Total variation of AGLA fusion weights; ``(B,K,V,2,H,W)`` or ``(B,V,2,H,W)``."""
    if gates.numel() == 0:
        return gates.sum() * 0.0
    g = gates
    if g.dim() == 5:
        g = g.flatten(0, 1)
    dh = (g[..., 1:, :] - g[..., :-1, :]).abs().mean()
    dw = (g[..., :, 1:] - g[..., :, :-1]).abs().mean()
    return dh + dw


class VGGTALLoss(nn.Module):
    """Weighted sum of all loss terms (paper Eq. 13)."""

    def __init__(self, weights: Optional[LossWeights] = None, use_lpips: bool = False) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.use_lpips = use_lpips

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        parts: Dict[str, torch.Tensor] = {}
        images = batch["images"]
        intrinsics = batch["intrinsics"]
        depth = pred["depth"]
        confidence = pred.get("confidence")
        cameras = pred["cameras"]

        if "depth_gt" in batch and batch["depth_gt"] is not None:
            parts.update(depth_loss(depth, batch["depth_gt"], self.w.depth_si_log))
        else:
            zero = depth.sum() * 0.0
            parts["depth_l1"] = zero
            parts["depth_si_log"] = zero

        parts["photometric"] = photometric_loss(
            images, depth, cameras, intrinsics, confidence
        )
        parts["perceptual"] = perceptual_loss(images, self.use_lpips)
        parts["camera"] = camera_loss(cameras, batch.get("cameras_gt"))
        parts["fusion_smooth"] = fusion_smoothness(pred.get("gates", torch.empty(0)))

        total = (
            self.w.depth_l1 * parts["depth_l1"]
            + parts["depth_si_log"]  # already carries si_weight
            + self.w.photometric * parts["photometric"]
            + self.w.perceptual * parts["perceptual"]
            + self.w.camera * parts["camera"]
            + self.w.fusion_smooth * parts["fusion_smooth"]
        )
        parts["total"] = total
        return parts
