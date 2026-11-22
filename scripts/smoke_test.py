#!/usr/bin/env python3
"""Dependency-light smoke test: synthetic data end-to-end, no downloads needed.

Checks
  1. AGLA forward shapes, gate normalization, all four fusion modes;
  2. VGGT-AL forward (depth / camera / point cloud heads);
  3. multi-task loss backpropagation and finite gradients;
  4. manifest / dataset loading with temporary synthetic PNGs;
  5. point-cloud metrics on known inputs.

Run: python scripts/smoke_test.py [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.attention import AGLA  # noqa: E402
from vggt_al.data import SparseViewSceneDataset  # noqa: E402
from vggt_al.losses import LossWeights, VGGTALLoss  # noqa: E402
from vggt_al.metrics import chamfer_distance, f_score  # noqa: E402
from vggt_al.model import VGGTAL, VGGTALConfig  # noqa: E402
from vggt_al.utils import load_config  # noqa: E402


def check_agla(device: torch.device) -> None:
    b, v, h, w, c = 2, 3, 8, 8, 64
    x = torch.randn(b, v * h * w, c, device=device)
    for mode in ("gated", "concat", "global_only", "local_only"):
        block = AGLA(c, num_heads=4, global_downsample=2, window_size=4,
                     mode=mode).to(device)
        for scope in ("intra", "cross"):
            y, gate = block(x, (h, w), v, scope=scope, return_gate=True)
            assert y.shape == x.shape, (mode, scope, y.shape)
            assert torch.isfinite(y).all(), (mode, scope)
            if mode == "gated":
                assert gate.shape == (b, v, 2, h, w)
                assert torch.allclose(gate.sum(dim=2), torch.ones_like(gate[:, :, 0]), atol=1e-6)
        y.sum().backward()
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    print("[ok] AGLA: 4 fusion modes x 2 scopes, shapes / gates / gradients")


def check_model(device: torch.device) -> None:
    cfg = VGGTALConfig(
        img_size=(64, 64), patch_size=8, embed_dim=64, depth=4,
        num_heads=4, num_views=3, global_downsample=2, window_size=4,
    )
    model = VGGTAL(cfg).to(device)
    b, v = 2, 3
    images = torch.rand(b, v, 3, 64, 64, device=device)
    intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).expand(b, v, -1, -1).clone()
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 40.0
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = 32.0
    out = model(images, intrinsics, return_gates=True)
    assert out["depth"].shape == (b, v, 1, 64, 64)
    assert out["cameras"].shape == (b, v, 4, 4)
    assert out["points"].shape == (b, v, 64, 64, 3)
    assert out["gates"].shape[0] == b
    assert torch.isfinite(out["points"]).all()
    assert torch.allclose(
        out["cameras"][..., :3, :3] @ out["cameras"][..., :3, :3].transpose(-1, -2),
        torch.eye(3, device=device).view(1, 1, 3, 3).expand(b, v, -1, -1),
        atol=2e-3,
    ), "predicted rotations must be orthogonal"

    criterion = VGGTALLoss(LossWeights(), use_lpips=False)
    batch = {
        "images": images,
        "intrinsics": intrinsics,
        "depth_gt": torch.rand(b, v, 1, 64, 64, device=device) * 5 + 1,
        "cameras_gt": None,
    }
    losses = criterion(out, batch)
    losses["total"].backward()
    for key, value in losses.items():
        assert torch.isfinite(value), key
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"[ok] VGGT-AL forward+backward: {model.num_parameters / 1e3:.1f}K params, "
          f"loss={losses['total'].item():.4f}")


def check_dataset() -> None:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = []
        for i in range(4):
            p = root / f"view_{i:02d}.png"
            Image.fromarray(np.full((48, 64, 3), 40 * i, dtype=np.uint8)).save(p)
            paths.append(str(p))
        manifest = {
            "dataset": "synthetic",
            "scenes": [{
                "name": "synthetic_000",
                "images": paths,
                "intrinsics": [[[40.0, 0, 32.0], [0, 40.0, 24.0], [0, 0, 1]]] * 4,
                "poses": None,
                "depth": None,
            }],
        }
        mp = root / "manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        ds = SparseViewSceneDataset(mp, img_size=(32, 48), num_views=3,
                                    augment=True, seed=1)
        item = ds[0]
        assert item["images"].shape == (3, 3, 32, 48)
        assert item["intrinsics"].shape == (3, 3, 3)
        assert torch.isfinite(item["intrinsics"]).all()
        assert item["depth_gt"].shape == (3, 1, 32, 48)
        print("[ok] dataset: manifest + 3-view sampling + shared augmentation")


def check_metrics() -> None:
    pts = np.random.default_rng(0).normal(size=(500, 3))
    assert abs(chamfer_distance(pts, pts)) < 1e-6
    assert abs(f_score(pts, pts, 0.05) - 1.0) < 1e-6
    drifted = pts.copy()
    drifted[:50] += np.array([0.2, 0.0, 0.0])  # partial shift -> partial match
    assert 0.0 < f_score(pts, drifted, 0.05) < 1.0
    assert chamfer_distance(pts, drifted) > 0.0
    print("[ok] metrics: CD / F-Score on identical and shifted clouds")


def check_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.json")
    assert cfg["model"]["window_size"] == 8 and cfg["model"]["global_downsample"] == 4
    print("[ok] config: default.json matches paper hyperparameters (w=8, s=4)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    check_config()
    check_agla(device)
    check_model(device)
    check_dataset()
    check_metrics()
    print("[PASS] all smoke tests completed")


if __name__ == "__main__":
    main()
