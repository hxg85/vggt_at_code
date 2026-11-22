#!/usr/bin/env python3
"""Show official download routes for the three datasets and verify local data.

This script deliberately does **not** mirror dataset files: all three datasets
have explicit licenses, and the official downloaders check MD5 / terms.  Use
``--print`` (default) for exact instructions, ``--verify`` to check a local
manifest, or ``--run-tnt-downloader`` to invoke the official Tanks and Temples
downloader with your arguments.

Examples
--------
python scripts/download_datasets.py --print
python scripts/download_datasets.py --verify manifests/dtu_test_6v.json
python scripts/download_datasets.py --run-tnt-downloader \
    --tnt_args "--modality image --group intermediate"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.data import SceneManifest  # noqa: E402

INFO = {
    "ss3dm": """\
SS3DM (NeurIPS 2024, Tsinghua THU-LYJ-Lab)
  project page : https://ss3dm.top/
  paper        : https://ss3dm.top/  (see linked PDF)
  exporter     : https://github.com/THU-LYJ-Lab/SS3DM-Exporter
  benchmark/GT : https://github.com/THU-LYJ-Lab/SS3DM-Benchmark
  download     : Google Drive / Hugging Face links on the project page
  content      : CARLA street scenes, 6 RGB cameras + 5 LiDAR, 13,535 frames,
                 precise synthetic meshes; 14 short + 8 middle + 6 long sequences.
  notes        : run SS3DM-Benchmark/extract_gt_pcd.py first to obtain the
                 reference point clouds, then scripts/prepare_ss3dm.py.
""",
    "dtu": """\
DTU MVS Dataset (CVPR 2014 / IJCV 2016, DTU Robot Image Data)
  official     : https://roboimagedata.compute.dtu.dk/?page_id=36
  direct files : http://roboimagedata.compute.dtu.dk/data/MVS/SampleSet.zip (6.3 GB)
                 http://roboimagedata.compute.dtu.dk/data/MVS/Points.zip    (6.3 GB)
  content      : 124 scenes, 49 or 64 calibrated views each, 7 lighting
                 conditions, 1600x1200 RGB, structured-light reference clouds.
  standard     : 15 test scans (24,37,40,55,63,65,69,83,97,105,106,110,114,118,122)
  notes        : use scripts/prepare_dtu.py to build the sparse-view manifest.
""",
    "tanks_and_temples": """\
Tanks and Temples Benchmark
  download     : https://www.tanksandtemples.org/download/
  toolbox      : https://github.com/isl-org/TanksAndTemples
  downloader   : python python_toolbox/download_t2_dataset.py \
                     --modality image --group intermediate
  content      : intermediate = 8 scenes (Family, Francis, Horse, Lighthouse,
                 M60, Panther, Playground, Train); advanced = 6 larger scenes.
  evaluation   : official F-score with scene-specific dTau; test GT withheld,
                 submit point clouds to the online server for official numbers.
  intrinsics   : pinhole prior fx=fy=0.7*W, cx=W/2, cy=H/2.
""",
}


def verify(manifest_path: str) -> None:
    manifest = SceneManifest.load(manifest_path)
    missing = 0
    total = 0
    for scene in manifest.scenes:
        files = list(scene.get("images") or [])
        if scene.get("depth"):
            files += scene["depth"]
        if scene.get("gt_pointcloud"):
            files.append(scene["gt_pointcloud"])
        for f in files:
            total += 1
            if not Path(f).is_file():
                missing += 1
                print(f"[missing] {scene['name']}: {f}")
    print(f"[verify] {manifest.dataset}: {total - missing}/{total} files present "
          f"across {len(manifest.scenes)} scenes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", dest="do_print", action="store_true")
    parser.add_argument("--verify", metavar="MANIFEST_JSON")
    parser.add_argument("--run-tnt-downloader", action="store_true")
    parser.add_argument("--tnt_args", default="--modality image --group intermediate")
    args = parser.parse_args()

    if args.do_print or not (args.verify or args.run_tnt_downloader):
        for text in INFO.values():
            print(text)
            print("-" * 72)
    if args.verify:
        verify(args.verify)
    if args.run_tnt_downloader:
        script = Path("TanksAndTemples/python_toolbox/download_t2_dataset.py")
        if not script.exists():
            print(f"[error] downloader not found at {script}; "
                  "clone https://github.com/isl-org/TanksAndTemples first")
            return
        cmd = [sys.executable, str(script)] + args.tnt_args.split()
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
