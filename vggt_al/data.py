"""Unified dataset adapter for SS3DM, DTU and Tanks and Temples.

All three datasets are converted by the scripts in ``scripts/`` into the same
manifest format (JSON), so training / evaluation code never needs to know
which benchmark it is reading::

    {
      "dataset": "dtu",
      "scenes": [
        {
          "name": "scan24",
          "images": [".../images/000000.png", ...],       # >= num_views
          "intrinsics": [[[fx,0,cx],[0,fy,cy],[0,0,1]], ...],
          "poses": [[[1,0,0,tx], ...], ...],               # optional c2w, else identity
          "depth": [".../depth/000000.npy", ...] | null,   # optional supervision
          "depth_scale": 1.0,
          "gt_pointcloud": ".../Points/scan24.ply" | null,
          "crop_json": null                                # Tanks and Temples
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


@dataclass
class SceneManifest:
    """Small wrapper around the JSON manifest with validation."""

    dataset: str = "unknown"
    scenes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "SceneManifest":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            raw = {"scenes": raw}
        scenes = raw.get("scenes", [])
        if not scenes:
            raise ValueError(f"no scenes found in manifest: {path}")
        required = ("name", "images")
        for i, scene in enumerate(scenes):
            for key in required:
                if key not in scene:
                    raise ValueError(f"scene #{i} missing required key '{key}'")
        return cls(
            dataset=raw.get("dataset", "unknown"),
            scenes=scenes,
            metadata=raw.get("metadata", {}),
        )

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"dataset": self.dataset, "metadata": self.metadata, "scenes": self.scenes},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def split(self, train_ratio: float = 0.8, seed: int = 0):
        """Deterministic scene-level split; returns two SceneManifest objects."""
        names = sorted(s["name"] for s in self.scenes)
        rng = random.Random(seed)
        rng.shuffle(names)
        n_train = int(len(names) * train_ratio)
        train_names, val_names = set(names[:n_train]), set(names[n_train:])
        train = SceneManifest(
            self.dataset,
            [s for s in self.scenes if s["name"] in train_names],
            dict(self.metadata, split="train"),
        )
        val = SceneManifest(
            self.dataset,
            [s for s in self.scenes if s["name"] in val_names],
            dict(self.metadata, split="val"),
        )
        return train, val


def _load_depth(path: str | os.PathLike, scale: float) -> np.ndarray:
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    elif path.endswith(".npz"):
        with np.load(path) as data:
            arr = np.asarray(data[data.files[0]], dtype=np.float32)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if np.nanmax(arr) > 255:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
    return arr * float(scale)


def read_pointcloud_ply(path: str | os.PathLike, max_points: int = 100_000) -> np.ndarray:
    """Read XYZ from a PLY, subsampling to ``max_points`` if needed.

    Uses ``plyfile`` when installed (handles binary PLY); falls back to a small
    ASCII parser for header-only binary files it cannot decode, in which case
    an informative error is raised.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        import plyfile  # type: ignore

        ply = plyfile.PlyData.read(path)
        vertex = ply["vertex"]
        xyz = np.stack(
            [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
            axis=-1,
        ).astype(np.float32)
    except ImportError:
        with open(path, "rb") as f:
            head = f.read(2048)
        if b"ascii" not in head.lower():
            raise RuntimeError(
                "Reading binary PLY requires the 'plyfile' package. "
                "Run: pip install plyfile"
            )
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines) and lines[i].strip() != "end_header":
            i += 1
        props = [ln.split()[-1] for ln in lines[:i] if ln.startswith("property")]
        ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
        xyz = np.array(
            [[float(v) for v in lines[j].split()] for j in range(i + 1, len(lines))],
            dtype=np.float32,
        )[:, [ix, iy, iz]]
    n = len(xyz)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        xyz = xyz[idx]
    return xyz


class SparseViewSceneDataset(Dataset):
    """PyTorch Dataset producing a fixed-size sparse-view batch item."""

    def __init__(
        self,
        manifest: SceneManifest | str | os.PathLike,
        img_size: Tuple[int, int] = (384, 512),
        num_views: int = 6,
        augment: bool = False,
        load_depth: bool = True,
        load_pointcloud: bool = False,
        max_points: int = 100_000,
        view_sampling: str = "first",
        seed: int = 0,
    ) -> None:
        self.manifest = (
            SceneManifest.load(manifest) if isinstance(manifest, (str, os.PathLike)) else manifest
        )
        self.img_size = tuple(img_size)  # (H, W)
        self.num_views = int(num_views)
        self.augment = augment
        self.load_depth = load_depth
        self.load_pointcloud = load_pointcloud
        self.max_points = max_points
        self.view_sampling = view_sampling
        self.rng_seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        for scene in self.manifest.scenes:
            if len(scene["images"]) < num_views:
                raise ValueError(
                    f"scene {scene['name']} has {len(scene['images'])} images, "
                    f"need >= {num_views}"
                )

    def __len__(self) -> int:
        return len(self.manifest.scenes)

    def _view_indices(self, n_available: int) -> List[int]:
        if n_available == self.num_views:
            return list(range(n_available))
        if self.view_sampling == "random":
            return sorted(self._rng.sample(range(n_available), self.num_views))
        return list(range(self.num_views))

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene = self.manifest.scenes[index]
        idxs = self._view_indices(len(scene["images"]))
        paths = [scene["images"][i] for i in idxs]
        images = [self._load_image(p) for p in paths]

        # Shared geometric augmentation across views (otherwise multi-view
        # consistency is destroyed); photometric jitter is per view.
        h, w = self.img_size
        flip = self.augment and self._rng.random() < 0.5
        crop: Optional[Tuple[int, int, int, int]] = None
        if self.augment:
            scale = self._rng.uniform(0.85, 1.0)
            cw, ch = int(w * scale), int(h * scale)
            x0 = self._rng.randint(0, w - cw)
            y0 = self._rng.randint(0, h - ch)
            crop = (x0, y0, x0 + cw, y0 + ch)

        tensors = []
        for im in images:
            if crop is not None:
                im = im.crop(crop)
            if flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            im = im.resize((w, h), Image.Resampling.BILINEAR)
            if self.augment:
                im = ImageEnhance.Color(im).enhance(self._rng.uniform(0.8, 1.2))
                im = ImageEnhance.Brightness(im).enhance(self._rng.uniform(0.9, 1.1))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        image_tensor = torch.stack(tensors)  # (V,3,H,W)

        # Intrinsics must follow cropping/resizing.  Original image size is
        # taken from the loaded PIL image so the manifest need not repeat it.
        orig_w, orig_h = images[0].size
        K = np.asarray(scene.get("intrinsics"), dtype=np.float32)
        if K.ndim == 2:
            K = np.stack([K] * self.num_views)
        K = K[idxs].copy()
        if crop is not None:
            x0, y0, x1, y1 = crop
            K[:, 0, 2] -= x0
            K[:, 1, 2] -= y0
            sx, sy = w / (x1 - x0), h / (y1 - y0)
        else:
            sx, sy = w / orig_w, h / orig_h
        K[:, 0, 0] *= sx
        K[:, 0, 2] *= sx
        K[:, 1, 1] *= sy
        K[:, 1, 2] *= sy
        if flip:
            K[:, 0, 2] = w - K[:, 0, 2]
        intrinsics = torch.from_numpy(K)

        raw_poses = scene.get("poses")
        if raw_poses is None:
            # Missing poses must stay None: identity placeholders would be
            # mistaken for real ground truth by the camera loss.
            cameras = None
        else:
            cameras = torch.as_tensor(
                np.asarray(raw_poses, dtype=np.float32)[idxs], dtype=torch.float32
            )

        depth_tensor = torch.zeros(self.num_views, 1, h, w, dtype=torch.float32)
        depth_paths = scene.get("depth")
        if self.load_depth and depth_paths:
            scale = float(scene.get("depth_scale", 1.0))
            ds = []
            for p in [depth_paths[i] for i in idxs]:
                arr = _load_depth(p, scale)
                if crop is not None:
                    arr = arr[crop[1]:crop[3], crop[0]:crop[2]]
                if flip:
                    arr = arr[:, ::-1]
                arr = np.asarray(
                    Image.fromarray(arr).resize((w, h), Image.Resampling.NEAREST),
                    dtype=np.float32,
                )
                ds.append(torch.from_numpy(arr).unsqueeze(0))
            depth_tensor = torch.stack(ds)

        out: Dict[str, Any] = {
            "images": image_tensor,
            "intrinsics": intrinsics,
            "cameras_gt": cameras,
            "depth_gt": depth_tensor if self.load_depth else None,
            "name": scene["name"],
        }
        if self.load_pointcloud and scene.get("gt_pointcloud"):
            xyz = read_pointcloud_ply(scene["gt_pointcloud"], self.max_points)
            out["points_gt"] = torch.from_numpy(xyz)
        else:
            out["points_gt"] = None
        return out
