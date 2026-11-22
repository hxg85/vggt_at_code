#!/usr/bin/env python3
"""Evaluate a trained VGGT-AL checkpoint on SS3DM / DTU / Tanks and Temples.

python scripts/evaluate.py --config configs/default.json \
    --manifest manifests/ss3dm_6v.json --checkpoint checkpoints/vggt_al/epoch_200.pt \
    --out outputs/ss3dm

Metrics follow the paper: Chamfer Distance, F-Score (threshold from config),
Normal Consistency, plus precision/recall.  For Tanks and Temples test scenes
GT is withheld, so only point-cloud export is produced -- submit the PLY files
to the official evaluation server for the official F-Score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.data import SparseViewSceneDataset  # noqa: E402
from vggt_al.metrics import evaluate_point_cloud  # noqa: E402
from vggt_al.model import VGGTAL, VGGTALConfig  # noqa: E402
from vggt_al.utils import (  # noqa: E402
    collate_sparse_view,
    load_checkpoint,
    load_config,
    save_point_cloud_ply,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--confidence_keep", type=float, default=0.6,
                        help="keep this fraction of predicted pixels by confidence")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    eval_cfg = dict(cfg["eval"])
    model_cfg = VGGTALConfig(**cfg["model"])
    ckpt = args.checkpoint or eval_cfg.get("checkpoint")
    manifest = args.manifest or eval_cfg.get("manifest")
    if not ckpt or not manifest:
        raise SystemExit("provide --checkpoint and --manifest")
    out_dir = Path(args.out or eval_cfg.get("output_dir", "outputs/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)

    model = VGGTAL(model_cfg).to(device).eval()
    load_checkpoint(ckpt, model, map_location=str(device))
    dataset = SparseViewSceneDataset(
        manifest=manifest,
        img_size=tuple(model_cfg.img_size),
        num_views=model_cfg.num_views,
        augment=False,
        load_depth=False,
        load_pointcloud=True,
        view_sampling="first",
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_sparse_view, num_workers=0,
    )

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["images"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            pred = model(images, intrinsics)
            points = pred["points"][0].reshape(-1, 3).cpu().numpy()  # (V*H*W, 3)
            conf = pred["confidence"][0].reshape(-1).cpu().numpy()
            keep = max(1, int(len(conf) * args.confidence_keep))
            sel = np.argpartition(conf, -keep)[-keep:]
            points = points[sel]
            name = batch["name"][0]
            if eval_cfg.get("export_ply", True):
                colors = None
                if "images" in batch:
                    rgb = batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3).numpy()
                    colors = np.asarray(rgb[sel] * 255, dtype=np.uint8)
                save_point_cloud_ply(out_dir / f"{name}.ply", points, colors)

            gt = batch.get("points_gt")
            row = {"scene": name, "points": len(points)}
            if gt is not None and gt[0] is not None:
                gt_np = gt[0].numpy().reshape(-1, 3)
                row.update(
                    evaluate_point_cloud(
                        points,
                        gt_np,
                        cd_threshold=float(eval_cfg.get("cd_threshold", 0.05)),
                        with_normals=bool(eval_cfg.get("with_normals", True)),
                        max_points=int(eval_cfg.get("max_points", 100_000)),
                    )
                )
                nc_text = f"NC={row['NC']:.4f}" if "NC" in row else "NC=disabled"
                print(f"[{i + 1}/{len(loader)}] {name}: "
                      f"CD={row['CD']:.4f} F={row['F-Score']:.4f} {nc_text}")
            else:
                print(f"[{i + 1}/{len(loader)}] {name}: exported PLY (no local GT)")
            rows.append(row)

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [
        r for r in rows if "CD" in r
    ]
    summary = {
        "scenes": len(rows),
        "scenes_with_gt": len(numeric),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    for key in ("CD", "F-Score", "NC", "Precision", "Recall"):
        if numeric and key in numeric[0]:
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in numeric]))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[done]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
