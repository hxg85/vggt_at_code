"""Point-cloud evaluation metrics used in the paper (Section 3.1).

* Chamfer Distance (CD): mean of the two one-way nearest-neighbour distances;
* F-Score at threshold ``t`` (t = 0.05 for SS3DM in the paper): harmonic mean
  of precision and recall of points within ``t`` of the reference cloud;
* Normal Consistency (NC): mean absolute cosine similarity between normals of
  matched points, in [0, 1], higher is better;
* Precision / Recall at a distance threshold (DTU, 1 mm in the paper).

The implementation uses ``scipy.spatial.cKDTree`` when available and falls
back to a chunked NumPy brute-force search otherwise.  All inputs are
``(N, 3)`` NumPy arrays, optionally sub-sampled for speed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _as_points(x: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if max_points is not None and len(arr) > max_points:
        idx = np.random.default_rng(0).choice(len(arr), max_points, replace=False)
        arr = arr[idx]
    if len(arr) == 0:
        raise ValueError("empty point cloud")
    return arr


def _one_way_distances(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance from every point in ``a`` to its nearest neighbour in ``b``."""
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(b)
        dist, _ = tree.query(a, k=1, workers=-1)
        return np.asarray(dist)
    except ImportError:
        out = []
        for start in range(0, len(a), chunk):
            block = a[start:start + chunk]
            d2 = ((block[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
            out.append(np.sqrt(d2.min(axis=1)))
        return np.concatenate(out)


def _nearest_indices(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(b)
        _, idx = tree.query(a, k=1, workers=-1)
        return np.asarray(idx)
    except ImportError:
        out = []
        for start in range(0, len(a), chunk):
            block = a[start:start + chunk]
            d2 = ((block[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
            out.append(d2.argmin(axis=1))
        return np.concatenate(out)


def chamfer_distance(
    pred: np.ndarray, gt: np.ndarray, max_points: Optional[int] = 100_000
) -> float:
    """Symmetric Chamfer distance (lower is better)."""
    p, g = _as_points(pred, max_points), _as_points(gt, max_points)
    return float(0.5 * (_one_way_distances(p, g).mean() + _one_way_distances(g, p).mean()))


def precision_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.05,
    max_points: Optional[int] = 100_000,
) -> Tuple[float, float]:
    """Precision / recall of points closer than ``threshold`` to the other set."""
    p, g = _as_points(pred, max_points), _as_points(gt, max_points)
    precision = float((_one_way_distances(p, g) < threshold).mean())
    recall = float((_one_way_distances(g, p) < threshold).mean())
    return precision, recall


def f_score(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.05,
    max_points: Optional[int] = 100_000,
) -> float:
    """F-score at distance threshold (higher is better)."""
    precision, recall = precision_recall(pred, gt, threshold, max_points)
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """PCA normals from local neighbourhoods; orientation is made consistent."""
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(points)
        _, idx = tree.query(points, k=min(k + 1, len(points)))
        idx = idx[:, 1:] if idx.shape[1] > 1 else idx
    except ImportError:
        d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
        idx = np.argsort(d2, axis=1)[:, 1:k + 1]
    nb = points[idx]  # (N, k, 3)
    centered = nb - points[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(centered.shape[1], 1)
    _, _, vh = np.linalg.svd(cov)
    normals = vh[:, -1, :]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    # Make normals point away from the global centroid (deterministic sign).
    to_center = points - points.mean(axis=0, keepdims=True)
    sign = np.sign(np.sum(normals * to_center, axis=1, keepdims=True))
    sign[sign == 0] = 1.0
    return normals * sign


def normal_consistency(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_normals: Optional[np.ndarray] = None,
    gt_normals: Optional[np.ndarray] = None,
    max_points: Optional[int] = 50_000,
) -> float:
    """Mean normal alignment between matched predicted / reference points."""
    p, g = _as_points(pred, max_points), _as_points(gt, max_points)
    pn = (
        estimate_normals(p)
        if pred_normals is None
        else np.asarray(pred_normals, dtype=np.float64).reshape(-1, 3)
    )
    gn = (
        estimate_normals(g)
        if gt_normals is None
        else np.asarray(gt_normals, dtype=np.float64).reshape(-1, 3)
    )
    if len(pn) != len(p) or len(gn) != len(g):
        raise ValueError("normals must have the same length as their point clouds")
    idx = _nearest_indices(p, g)
    dots = np.abs(np.sum(pn * gn[idx], axis=1))
    return float(dots.mean())


def evaluate_point_cloud(
    pred: np.ndarray,
    gt: np.ndarray,
    cd_threshold: float = 0.05,
    with_normals: bool = True,
    max_points: Optional[int] = 100_000,
) -> Dict[str, float]:
    """Convenience wrapper returning CD / F-Score / P / R / NC."""
    result: Dict[str, float] = {
        "CD": chamfer_distance(pred, gt, max_points),
    }
    p, r = precision_recall(pred, gt, cd_threshold, max_points)
    result["Precision"] = p
    result["Recall"] = r
    result["F-Score"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    if with_normals:
        result["NC"] = normal_consistency(pred, gt, max_points=min(50_000, max_points or 50_000))
    return result
