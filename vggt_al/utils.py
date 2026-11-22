"""Small utilities: config, seeding, checkpointing, PLY export, collate."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load JSON config; YAML is supported when PyYAML is installed."""
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML: pip install PyYAML") from exc
    return json.loads(text)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "config": getattr(model, "cfg", None),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | os.PathLike, model: torch.nn.Module, map_location: str = "cpu"
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[checkpoint] {len(missing)} keys not found in checkpoint (e.g. {missing[:3]})")
    if unexpected:
        print(f"[checkpoint] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return payload


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def collate_sparse_view(batch):
    """Collate dicts containing ``None`` values (e.g. missing depth / GT cloud)."""
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if any(v is None for v in values):
            out[key] = None
        elif isinstance(values[0], str):
            out[key] = values
        elif torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def save_point_cloud_ply(
    path: str | os.PathLike,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Write a minimal ASCII PLY (works with MeshLab / Open3D / CloudCompare)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)
        if colors.shape[0] != pts.shape[0]:
            raise ValueError("colors must match points")
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            line = f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
            if colors is not None:
                c = colors[i]
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")
