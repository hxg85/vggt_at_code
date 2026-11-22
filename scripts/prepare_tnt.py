#!/usr/bin/env python3
"""Convert Tanks and Temples into the VGGT-AL sparse-view manifest.

Official download / evaluation: https://www.tanksandtemples.org/download/
Toolbox (evaluation + downloader): https://github.com/isl-org/TanksAndTemples

Intermediate set (used in the paper): Family, Francis, Horse, Lighthouse,
M60, Panther, Playground, Train.  The downloader's
``--modality image --group intermediate`` option gives uniformly sampled
frames, which is the recommended quick-start input.

Note on ground truth: the *test* intermediate set withholds laser scans; local
F-score evaluation is only possible on the *training* scenes (Barn,
Caterpillar, Church, Courthouse, Ignatius, Meeting room, Truck) or via an
online submission.  This script fills ``gt_pointcloud`` only when a PLY is
found locally, and always records the ``crop_json`` path for the official
evaluator.

Example
-------
python scripts/prepare_tnt.py --data_root /data/TanksAndTemples \
    --group intermediate --num_views 6 --out manifests/tnt_intermediate_6v.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.data import SceneManifest  # noqa: E402

INTERMEDIATE = [
    "Family", "Francis", "Horse", "Lighthouse",
    "M60", "Panther", "Playground", "Train",
]
TRAINING = [
    "Barn", "Caterpillar", "Church", "Courthouse",
    "Ignatius", "Meeting room", "Truck",
]


def _find_images(scene_dir: Path) -> List[Path]:
    folders = [scene_dir, scene_dir / "images", scene_dir / "image", scene_dir / "frames"]
    for folder in folders:
        if folder.is_dir():
            files = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file()
            )
            if len(files) >= 3:
                return files
    raise FileNotFoundError(f"no image sequence found under {scene_dir}")


def _find_gt(scene_dir: Path) -> Optional[str]:
    candidates = [
        scene_dir / "gt.ply",
        scene_dir / "ground_truth.ply",
        scene_dir / "groundtruth.ply",
        scene_dir / "points.ply",
        *sorted(scene_dir.rglob("*gt*.ply")),
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def _find_crop(scene_dir: Path) -> Optional[str]:
    for p in [*scene_dir.glob("*.json"), *scene_dir.rglob("*crop*.json")]:
        if "crop" in p.name.lower():
            return str(p.resolve())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--group", choices=["intermediate", "training", "advanced"], default="intermediate")
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--intrinsics_focal_ratio", type=float, default=0.7,
                        help="f = ratio * image width (official pinhole prior)")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    names = {
        "intermediate": INTERMEDIATE,
        "training": TRAINING,
        "advanced": ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"],
    }[args.group]

    scenes = []
    for name in names:
        # Accept both flat (<root>/Family) and grouped (<root>/intermediate/Family)
        candidates = [root / name, root / args.group / name, root / "train" / name]
        scene_dir = next((p for p in candidates if p.is_dir()), None)
        if scene_dir is None:
            print(f"[skip] scene not found: {name}")
            continue
        try:
            files = _find_images(scene_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}")
            continue
        idx = sorted(set(int(round(i * (len(files) - 1) / max(args.num_views - 1, 1)))
                         for i in range(args.num_views)))
        while len(idx) < args.num_views:
            idx.append(min(idx[-1] + 1, len(files) - 1))
        selected = [files[i] for i in idx]

        # Image size is read from the first frame to build the pinhole prior.
        from PIL import Image

        with Image.open(selected[0]) as im:
            w, h = im.size
        f = args.intrinsics_focal_ratio * w
        K = [[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]]
        gt = _find_gt(scene_dir)
        crop = _find_crop(scene_dir)
        if args.group != "training" and gt is None:
            print(f"[info] {name}: no local GT (expected for the test set); "
                  "use the official evaluation server")

        scenes.append(
            {
                "name": name,
                "images": [str(p.resolve()) for p in selected],
                "intrinsics": [K] * len(selected),
                "poses": None,  # feed-forward model predicts cameras
                "depth": None,
                "gt_pointcloud": gt,
                "crop_json": crop,
                "frame_indices": idx,
            }
        )

    manifest = SceneManifest(
        dataset="tanks_and_temples",
        scenes=scenes,
        metadata={
            "group": args.group,
            "num_views": args.num_views,
            "source": "https://www.tanksandtemples.org/download/",
            "evaluator": "https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation",
            "note": "Test-set ground truth is withheld; report F-score via the "
                    "official server unless local GT is present.",
        },
    )
    manifest.save(args.out)
    print(f"[TnT] wrote {len(scenes)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
