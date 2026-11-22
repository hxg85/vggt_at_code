#!/usr/bin/env python3
"""Convert SS3DM (NeurIPS 2024) into the VGGT-AL sparse-view manifest.

Official sources
----------------
* project page / paper: https://ss3dm.top/
* exporter (CARLA data format): https://github.com/THU-LYJ-Lab/SS3DM-Exporter
* benchmark + GT extraction: https://github.com/THU-LYJ-Lab/SS3DM-Benchmark
* data download: Google Drive / Hugging Face links on the project page

The exporter writes one directory per sequence with ``images/``, ``depths/``,
``poses/``, ``lidars/`` etc.  A vehicle carries six RGB cameras, so six images
sharing the same frame index form one sparse-view sample -- exactly the
6-view setup used in the paper.

Ground-truth point clouds are not shipped with the raw export: run
``SS3DM-Benchmark/extract_gt_pcd.py --data_root <root>`` first.  It produces
``selected_gt_point_cloud_dense_resampled.ply`` in each sequence directory.
This script picks that file up automatically when present.

IMPORTANT: the paper reports an 800-train / 200-test scene split but does not
publish the exact scene IDs.  This script therefore provides a deterministic
split (``--split_mode scenes`` default) and records it in the manifest
metadata.  To reproduce the published numbers exactly you must use the
authors' split; otherwise treat resulting numbers as an independent estimate.

Example
-------
python scripts/prepare_ss3dm.py --data_root /data/SS3DM \
    --out manifests/ss3dm_6v.json --split_mode scenes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.data import SceneManifest  # noqa: E402

DEFAULT_GT = "selected_gt_point_cloud_dense_resampled.ply"


def _frame_id(name: str, pattern: re.Pattern) -> Optional[int]:
    matches = pattern.findall(Path(name).stem)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _group_by_frame(folder: Path, pattern: re.Pattern, exts: Tuple[str, ...]):
    groups = defaultdict(list)
    if not folder.is_dir():
        return groups
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            fid = _frame_id(p.name, pattern)
            if fid is not None:
                groups[fid].append(p)
    return groups


def _load_pose(path: Path) -> Optional[List[float]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for key in ("c2w", "camera_to_world", "pose", "transform", "matrix"):
                if key in obj:
                    mat = np.asarray(obj[key], dtype=np.float32)
                    break
            else:
                mat = np.asarray(obj, dtype=np.float32)
        else:
            mat = np.asarray(obj, dtype=np.float32)
        mat = mat.reshape(4, 4)
        return mat.tolist()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--frame_regex", default=r"(\d+)",
                        help="regex whose LAST group is the frame index")
    parser.add_argument("--gt_name", default=DEFAULT_GT)
    parser.add_argument("--split_mode", choices=["scenes", "sequences", "none"], default="scenes")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    pattern = re.compile(args.frame_regex)
    # A sequence is any directory that directly contains an images/ folder.
    seq_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and (p / "images").is_dir())
    # Keep only the shallowest matches when nested layouts repeat.
    seq_dirs = [d for d in seq_dirs if not any(x != d and d in x.parents for x in seq_dirs)]
    print(f"[SS3DM] found {len(seq_dirs)} sequences")

    scenes: List[Dict] = []
    for seq in seq_dirs:
        images = _group_by_frame(seq / "images", pattern, (".png", ".jpg", ".jpeg"))
        depths = _group_by_frame(seq / "depths", pattern, (".png", ".npy", ".exr", ".npz"))
        pose_folder = seq / "poses"
        gt_path = seq / args.gt_name
        if not gt_path.exists():
            for candidate in seq.rglob(DEFAULT_GT):
                gt_path = candidate
                break
        gt = str(gt_path.resolve()) if gt_path.exists() else None

        for fid in sorted(images):
            paths = sorted(images[fid])[: args.num_views]
            if len(paths) < args.num_views:
                continue
            # Intrinsics: SS3DM's RGB cameras are calibrated in CARLA.  If a
            # calibration.json exists next to the sequence, use it; otherwise
            # fall back to the widely used pinhole prior (f = 0.7 W).
            K = None
            for calib in (seq / "calibration.json", root / "calibration.json"):
                if calib.exists():
                    try:
                        obj = json.loads(calib.read_text(encoding="utf-8"))
                        candidate = obj.get("intrinsic") if isinstance(obj, dict) else None
                        if candidate is not None:
                            K = np.asarray(candidate, dtype=np.float32).reshape(3, 3).tolist()
                            break
                    except Exception:
                        pass
            if K is None:
                from PIL import Image

                with Image.open(paths[0]) as im:
                    w, h = im.size
                K = [[0.7 * w, 0.0, 0.5 * w], [0.0, 0.7 * w, 0.5 * h], [0.0, 0.0, 1.0]]

            poses = None
            if pose_folder.is_dir():
                candidates = sorted(pose_folder.glob(f"*{fid}*"))
                if not candidates:
                    all_poses = pose_folder / "poses.json"
                    if all_poses.exists():
                        try:
                            obj = json.loads(all_poses.read_text(encoding="utf-8"))
                            poses = obj[fid] if isinstance(obj, dict) and fid in obj else None
                        except Exception:
                            poses = None
                else:
                    loaded = [_load_pose(p) for p in candidates[: args.num_views]]
                    if all(p is not None for p in loaded):
                        poses = loaded

            depth_paths = [str(p.resolve()) for p in depths.get(fid, [])[: args.num_views]]
            if len(depth_paths) != args.num_views:
                depth_paths = None
            scenes.append(
                {
                    "name": f"{seq.name}_{fid:06d}",
                    "sequence": seq.name,
                    "frame": fid,
                    "images": [str(p.resolve()) for p in paths],
                    "intrinsics": [K] * len(paths),
                    "poses": poses,
                    "depth": depth_paths,
                    "depth_scale": 1.0,
                    "gt_pointcloud": gt,
                    "crop_json": None,
                }
            )

    manifest = SceneManifest(
        dataset="ss3dm",
        scenes=scenes,
        metadata={
            "num_views": args.num_views,
            "frame_regex": args.frame_regex,
            "source": "https://ss3dm.top/",
            "gt_extractor": "https://github.com/THU-LYJ-Lab/SS3DM-Benchmark",
            "note": "Paper reports 800/200 train/test but does not publish scene IDs; "
                    "this manifest uses a deterministic split.",
        },
    )
    if args.split_mode == "scenes":
        train, val = manifest.split(args.train_ratio, args.seed)
        manifest.metadata["split"] = f"scenes:{args.train_ratio}:{args.seed}"
        # Mark split membership while keeping all scenes in one manifest.
        val_names = {s["name"] for s in val.scenes}
        for s in manifest.scenes:
            s["split"] = "val" if s["name"] in val_names else "train"
    elif args.split_mode == "sequences":
        names = sorted({s["sequence"] for s in manifest.scenes})
        cutoff = int(len(names) * args.train_ratio)
        val_seqs = set(names[cutoff:])
        for s in manifest.scenes:
            s["split"] = "val" if s["sequence"] in val_seqs else "train"
        manifest.metadata["split"] = f"sequences:{args.train_ratio}"
    else:
        for s in manifest.scenes:
            s["split"] = "all"
        manifest.metadata["split"] = "none"

    manifest.save(args.out)
    n_train = sum(s["split"] == "train" for s in manifest.scenes)
    print(f"[SS3DM] wrote {len(scenes)} scenes ({n_train} train) -> {args.out}")


if __name__ == "__main__":
    main()
