# VGGT-AL

**Adaptive Global-Local Attention for Large-Scale 3D Scene Reconstruction from Sparse Views**

This repository contains the PyTorch implementation of VGGT-AL, a framework for **3D reconstruction of large-scale scenes from 4–6 sparse views**. Its core contribution is **AGLA (Adaptive Global-Local Attention)**: the standard self-attention in the VGGT encoder is replaced by two parallel branches -- global attention on downsampled features and local window attention at the original resolution -- combined by a learnable, content-adaptive gate. This design preserves global geometric consistency while retaining fine local details. The repository also provides data preparation, training, and evaluation scripts for three benchmarks: SS3DM, DTU, and Tanks and Temples.

---

## 1. Method

| Component | Location | Description |
|---|---|---|
| Global branch | `vggt_al/attention.py` → `MultiHeadSelfAttention` and the global branch of `AGLA.forward` | Average-pool the feature map with stride `s=4`, apply multi-head self-attention, then bilinearly upsample back to the original resolution |
| Local branch | `WindowSelfAttention` | Non-overlapping `8×8` window self-attention at the original resolution, with relative position bias and support for arbitrary resolutions via padding |
| Adaptive fusion | `AGLA.gate_conv` | A lightweight 3×3 convolution predicts two channels, normalized by softmax into spatially varying weights `G ∈ R^(B,V,2,H,W)`; initialized to equal weights |
| Encoder block | `AGLABlock` + `VGGTAL` | Alternating frame-wise and cross-view attention, retaining positional encoding and the feed-forward network |
| Multi-task loss | `vggt_al/losses.py` → `VGGTALLoss` | Depth regression + radiance-field reconstruction + camera-pose consistency + fusion-weight smoothing |
| Outputs | `VGGTAL.forward` | `depth / confidence / cameras / points` (depth maps, confidence, camera poses, 3D point clouds) |

The AGLA module computes:

```
Xg = Upsample( MHSA( LN( AvgPool2d(X, stride=s=4) ) ) )   # cross-view or per-view, determined by the block scope
Xl = WindowMHSA( LN(X), window=8 )                        # per view
G  = softmax( Conv3x3(X), dim=channel )                   # two spatial weight maps summing to one
Y  = Linear( G0 ⊙ Xg + G1 ⊙ Xl )
```

Ablation variants are selected through `config.model.fusion_mode`: `gated` (full method), `concat` (fixed concatenation), `global_only`, and `local_only`, corresponding to Table 2 of the paper.

---

## 2. Datasets

| Dataset | Source | Scale and protocol | Metrics |
|---|---|---|---|
| **SS3DM** | Project page <https://ss3dm.top/>; exporter [SS3DM-Exporter](https://github.com/THU-LYJ-Lab/SS3DM-Exporter); benchmark and GT extraction [SS3DM-Benchmark](https://github.com/THU-LYJ-Lab/SS3DM-Benchmark) | Synthetic street-view dataset built with CARLA; 6 RGB cameras + 5 LiDAR sensors, 13,535 frames; 14 short + 8 middle + 6 long sequences. Run `extract_gt_pcd.py` first to generate reference point clouds | CD, F-Score (t=0.05), Normal Consistency |
| **DTU MVS** | Official page <https://roboimagedata.compute.dtu.dk/?page_id=36>; `SampleSet.zip` (images + calibration), `Points.zip` (structured-light reference clouds) | 124 scenes, 49 or 64 calibrated views each under 7 lighting conditions, 1600×1200; standard 15 test scans (24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122); 3-view sets `[23,24,33]` and `[42,43,44]`, extensible to 4/6 views by the provided scripts | CD, Precision, Recall (1 mm threshold) |
| **Tanks and Temples** | Download page <https://www.tanksandtemples.org/download/>; toolbox [TanksAndTemples](https://github.com/isl-org/TanksAndTemples) | Intermediate set of 8 scenes (Family / Francis / Horse / Lighthouse / M60 / Panther / Playground / Train); Advanced set of 6 scenes; Training set of 7 scenes. Ground truth for the test sets is withheld and evaluated online | F-Score (scene-specific dTau) |

Run `python scripts/download_datasets.py --print` for download instructions and `--verify <manifest.json>` to check local files.

---

## 3. Repository Structure

```
.
├── README.md
├── requirements.txt
├── configs/default.json         model and training hyper-parameters (w=8, s=4, AdamW 1e-4, cosine, 512×384…)
├── vggt_al/                     core package
│   ├── attention.py             AGLA / AGLABlock / window attention
│   ├── model.py                 VGGT-AL encoder-decoder, depth/camera heads, point-cloud backprojection
│   ├── losses.py                multi-task losses
│   ├── data.py                  unified manifest dataset interface
│   ├── metrics.py               CD / F-Score / NC / Precision / Recall
│   └── utils.py                 config, checkpointing, PLY export, etc.
├── scripts/
│   ├── download_datasets.py     download instructions and local verification
│   ├── prepare_ss3dm.py         SS3DM → manifest
│   ├── prepare_dtu.py           DTU → manifest (3/4/6 views)
│   ├── prepare_tnt.py           Tanks and Temples → manifest
│   ├── train.py                 training
│   ├── evaluate.py              inference + metrics + PLY export
│   └── smoke_test.py            end-to-end self-check (no data download required)
└── tests/                       unit tests
```

The `manifest` is a JSON index shared by all three datasets. It stores, for each scene, the image paths, `3×3` intrinsics, optional `4×4` camera extrinsics, optional depth maps, and an optional ground-truth point-cloud path.

---

## 4. Quick Start

```bash
# Environment
pip install -r requirements.txt

# Self-check (no data download required)
python scripts/smoke_test.py
python -m unittest discover -s tests -v

# Download instructions
python scripts/download_datasets.py --print

# Data preparation (replace paths with your own)
python scripts/prepare_dtu.py --data_root /data/DTU --split train --num_views 6 \
    --out manifests/dtu_train_6v.json
python scripts/prepare_dtu.py --data_root /data/DTU --split test --num_views 6 \
    --out manifests/dtu_test_6v.json
python scripts/prepare_ss3dm.py --data_root /data/SS3DM --out manifests/ss3dm_6v.json
python scripts/prepare_tnt.py --data_root /data/TanksAndTemples \
    --group intermediate --out manifests/tnt_6v.json

# Training / evaluation
python scripts/train.py --config configs/default.json \
    --manifest manifests/dtu_train_6v.json --out checkpoints/dtu_al
python scripts/evaluate.py --config configs/default.json \
    --manifest manifests/dtu_test_6v.json \
    --checkpoint checkpoints/dtu_al/epoch_200.pt --out outputs/dtu_al
```

In `configs/default.json`, `epochs=200`, `lr=1e-4`, `weight_decay=1e-2`, and the cosine schedule follow the paper. The default `batch_size=8` targets single-GPU usage; increase it to 32 for the multi-GPU setup. Use `--max-steps 20` for a quick check of the training loop.

`VGGTAL.init_from_vggt(state_dict)` loads pretrained VGGT attention weights into the global branches; the local branches and fusion gates keep their paper-specified initialization so the pretrained representation is not disturbed.

---

## 5. Implementation Notes

1. **Model size**: The default configuration is a compact setting for single-GPU use (`embed_dim=384, depth=12`). The number of layers and feature dimensions can be changed in `configs/default.json`.
2. **SS3DM split**: `prepare_ss3dm.py` provides a deterministic scene-level split, recorded in the manifest metadata (`split` field) for reproducible experiments.
3. **Tanks and Temples evaluation**: Ground truth for the test sets is withheld, so local inference only exports PLY files; submit them to the official evaluation server for the official F-Score. `eval.cd_threshold` defaults to 0.05 (SS3DM); use 0.001 (1 mm) for DTU.
4. **CD units**: `metrics.chamfer_distance` returns distances in the native coordinate units of the point clouds; unify scene scale before cross-dataset comparison.
5. **Camera-pose supervision**: The `camera` loss is enabled when `poses` are provided in the manifest; otherwise it is disabled automatically and camera geometry is supervised by the photometric reprojection loss.
6. **Optional dependencies**: Without `scipy`, metrics fall back to chunked NumPy nearest-neighbour search (identical results, slower). Without `lpips`, the perceptual term uses a differentiable edge prior; install `lpips` and set `loss.use_lpips=true` to use LPIPS.

---

## 6. Citation

If you use this code, please cite:

```
Adaptive Global-Local Attention for Large-Scale 3D Scene Reconstruction from Sparse Views
```

Please also cite the datasets: SS3DM (NeurIPS 2024), DTU MVS (CVPR 2014 / IJCV 2016), and Tanks and Temples (CVPR 2017).
