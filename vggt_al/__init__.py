"""VGGT-AL: Adaptive Global-Local Attention for sparse-view 3D reconstruction.

Reference paper: "Adaptive Global-Local Attention for Large-Scale 3D Scene
Reconstruction from Sparse Views" (VGGT-AL, this repository's paper).

The package is intentionally self-contained: it implements the proposed AGLA
module in pure PyTorch, a lightweight VGGT-style encoder/decoder that accepts
AGLA as a drop-in replacement for self-attention, multi-task losses, dataset
adapters and evaluation metrics for SS3DM / DTU / Tanks and Temples.
"""

from .attention import AGLA, AGLABlock, MultiHeadSelfAttention, WindowSelfAttention
from .model import VGGTAL, VGGTALConfig
from .losses import VGGTALLoss, LossWeights
from .data import SparseViewSceneDataset, SceneManifest
from .metrics import (
    chamfer_distance,
    f_score,
    normal_consistency,
    precision_recall,
    evaluate_point_cloud,
)

__all__ = [
    "AGLA",
    "AGLABlock",
    "MultiHeadSelfAttention",
    "WindowSelfAttention",
    "VGGTAL",
    "VGGTALConfig",
    "VGGTALLoss",
    "LossWeights",
    "SparseViewSceneDataset",
    "SceneManifest",
    "chamfer_distance",
    "f_score",
    "normal_consistency",
    "precision_recall",
    "evaluate_point_cloud",
]

__version__ = "0.1.0"
