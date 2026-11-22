#!/usr/bin/env python3
"""Convert the DTU MVS dataset into the VGGT-AL sparse-view manifest.

Official data: https://roboimagedata.compute.dtu.dk/?page_id=36
Download ``SampleSet`` (6.3 GB, contains images + calibration) and ``Points``
(6.3 GB, structured-light reference clouds) and unpack them into one root.

Standard protocol used by learned sparse-view methods (MVSNet / pixelNeRF /
SparseNeuS ...): 15 test scans
    24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122
For surface reconstruction the common 3-view sets are
    set0 = [23, 24, 33],   set1 = [42, 43, 44]
This script extends those sets to 4 or 6 views by appending
    [25, 34, 45] and [26, 35, 46] (paper Section 3.1 uses 3 / 4 / 6 views).

Example
-------
python scripts/prepare_dtu.py --data_root /data/DTU --split test \
    --num_views 6 --out manifests/dtu_test_6v.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.data import SceneManifest  # noqa: E402

TEST_SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
VIEW_SETS = {
    3: [[23, 24, 33], [42, 43, 44]],
    4: [[23, 24, 33, 25], [42, 43, 44, 34]],
    6: [[23, 24, 33, 25, 26, 35], [42, 43, 44, 34, 45, 46]],
}


def _find_image_dir(scan: Path) -> Path:
    for name in ("image", "images", "rect", "rectified"):
        candidate = scan / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no image directory under {scan}")


def _image_path(image_dir: Path, view_id: int) -> Path:
    direct = image_dir / f"{view_id:08d}.png"
    if direct.exists():
        return direct
    pattern = re.compile(rf"(?:^|[_\-])0*{view_id}(?:[_\.]|$)")
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg") and pattern.search(p.name):
            return p
    raise FileNotFoundError(f"view {view_id} not found in {image_dir}")


def _load_calibration(root: Path) -> Dict[int, Dict]:
    for candidate in (root / "calibration.json", root / "SampleSet" / "calibration.json"):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                calib = json.load(f)
            if isinstance(calib, list):
                calib = {int(entry["scan"]): entry for entry in calib}
            return {int(k): v for k, v in calib.items()}
    return {}


def _scan_dirs(root: Path) -> List[Path]:
    dirs = []
    for base in (root, root / "SampleSet", root / "Rectified", root / "rectified"):
        if base.is_dir():
            for p in sorted(base.glob("scan*")):
                if p.is_dir() and _safe_int(p.name[4:]) is not None:
                    dirs.append(p)
    # de-duplicate by scan id, prefer the first occurrence
    seen = set()
    unique = []
    for d in dirs:
        sid = int(d.name[4:])
        if sid not in seen:
            seen.add(sid)
            unique.append(d)
    return sorted(unique, key=lambda p: int(p.name[4:]))


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None


def _find_gt(root: Path, scan_id: int) -> Optional[str]:
    for candidate in (
        root / "Points" / f"scan{scan_id:03d}.ply",
        root / "Points" / "stl" / f"pcd{scan_id:04d}.ply",
        root / "Points" / f"scan{scan_id}.ply",
        root / "Points_MVS" / f"scan{scan_id:03d}.ply",
    ):
        if candidate.exists():
            return str(candidate.resolve())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="root containing SampleSet/, Points/")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--num_views", type=int, choices=[3, 4, 6], default=6)
    parser.add_argument("--out", required=True, help="output manifest .json")
    parser.add_argument("--view_set", type=int, choices=[0, 1], default=0)
    parser.add_argument("--skip_missing_gt", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    calib = _load_calibration(root)
    scenes = []
    for scan_dir in _scan_dirs(root):
        scan_id = int(scan_dir.name[4:])
        if args.split == "test" and scan_id not in TEST_SCANS:
            continue
        if args.split == "train" and scan_id in TEST_SCANS:
            continue
        try:
            image_dir = _find_image_dir(scan_dir)
            image_paths = [
                _image_path(image_dir, vid) for vid in VIEW_SETS[args.num_views][args.view_set]
            ]
        except FileNotFoundError as exc:
            print(f"[skip] {scan_dir.name}: {exc}")
            continue
        gt = _find_gt(root, scan_id)
        if gt is None and not args.skip_missing_gt:
            print(f"[warn] {scan_dir.name}: no reference point cloud found")

        entry = calib.get(scan_id, {})
        K = np.asarray(entry.get("intrinsic", np.eye(3)), dtype=np.float32)
        if K.shape != (3, 3):
            K = np.eye(3, dtype=np.float32)
        ext = entry.get("extrinsic")
        poses = None
        if ext is not None:
            # DTU stores world-to-camera [R|t]; invert to camera-to-world.
            ext = np.asarray(ext, dtype=np.float64).reshape(3, 4)
            w2c = np.eye(4)
            w2c[:3] = ext
            c2w = np.linalg.inv(w2c)
            poses = [c2w.tolist()] * len(image_paths)
        scenes.append(
            {
                "name": f"scan{scan_id:03d}",
                "images": [str(p.resolve()) for p in image_paths],
                "intrinsics": [K.tolist()] * len(image_paths),
                "poses": poses,
                "depth": None,
                "depth_scale": 1.0,
                "gt_pointcloud": gt,
                "crop_json": None,
                "view_ids": VIEW_SETS[args.num_views][args.view_set],
            }
        )

    manifest = SceneManifest(
        dataset="dtu",
        scenes=scenes,
        metadata={
            "split": args.split,
            "num_views": args.num_views,
            "view_set": args.view_set,
            "source": "https://roboimagedata.compute.dtu.dk/?page_id=36",
            "note": "CD/Precision/Recall are evaluated on masked point clouds; "
                    "the official 1mm threshold is used for DTU.",
        },
    )
    manifest.save(args.out)
    print(f"[DTU] wrote {len(scenes)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
