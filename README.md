# VGGT-AL — Official Implementation

**Adaptive Global-Local Attention for Large-Scale 3D Scene Reconstruction from Sparse Views**

本仓库是 VGGT-AL 的官方 PyTorch 实现。VGGT-AL 面向 **4–6 张稀疏图像的大场景三维重建**，核心贡献是 **AGLA（Adaptive Global-Local Attention）**：用两条并行分支（降采样后的全局注意力 + 原分辨率窗口局部注意力）与可学习的内容自适应门控，替换 VGGT 编码器中的标准自注意力，在保持全局几何一致性的同时保留局部细节。仓库同时提供 SS3DM / DTU / Tanks and Temples 三个数据集的准备、训练与评测脚本。

---

## 1. 方法

| 模块 | 实现位置 | 说明 |
|---|---|---|
| 全局分支 | `vggt_al/attention.py` → `MultiHeadSelfAttention` + `AGLA.forward` 的 global 分支 | 特征图按 `s=4` 平均池化后做多头自注意力，再双线性上采样回原分辨率 |
| 局部分支 | `WindowSelfAttention` | 原分辨率下 `8×8` 非重叠窗口内自注意力，带相对位置偏置，支持任意分辨率 padding |
| 自适应融合 | `AGLA.gate_conv` | 轻量卷积输出 2 通道、softmax 归一化，得到空间自适应权重 `G ∈ R^(B,V,2,H,W)`；初始权重均等 |
| 编码器 Block | `AGLABlock` + `VGGTAL` | 帧内 / 跨视图注意力交替，保留位置编码与 FFN |
| 多任务损失 | `vggt_al/losses.py` → `VGGTALLoss` | 深度回归 + 辐射场重建 + 相机位姿约束 + 融合权重平滑 |
| 输出 | `VGGTAL.forward` | `depth / confidence / cameras / points`（深度图、置信度、相机位姿、三维点云） |

AGLA 的计算流程：

```
Xg = Upsample( MHSA( LN( AvgPool2d(X, stride=s=4) ) ) )   # 跨视图或单视图，取决于 block 的 scope
Xl = WindowMHSA( LN(X), window=8 )                        # 逐视图
G  = softmax( Conv3x3(X), dim=channel )                   # 两个空间权重图，和为 1
Y  = Linear( G0 ⊙ Xg + G1 ⊙ Xl )
```

消融配置通过 `config.model.fusion_mode` 切换：`gated`（完整方法）、`concat`（固定拼接）、`global_only`、`local_only`，对应论文 Table 2。

---

## 2. 实验数据集

| 数据集 | 来源 | 规模与协议 | 评测指标 |
|---|---|---|---|
| **SS3DM** | 项目主页 <https://ss3dm.top/>；导出器 [SS3DM-Exporter](https://github.com/THU-LYJ-Lab/SS3DM-Exporter)；评测与 GT 提取 [SS3DM-Benchmark](https://github.com/THU-LYJ-Lab/SS3DM-Benchmark) | 基于 CARLA 的合成街景数据集，6 路 RGB + 5 路 LiDAR，共 13,535 帧；14 short + 8 middle + 6 long 序列。先用 `extract_gt_pcd.py` 生成参考点云 | CD、F-Score（t=0.05）、Normal Consistency |
| **DTU MVS** | 官方页 <https://roboimagedata.compute.dtu.dk/?page_id=36>；`SampleSet.zip`（图像 + 标定）、`Points.zip`（结构光参考点云） | 124 个场景，每场景 49 或 64 视角、7 种光照，1600×1200；标准 15 个测试 scan（24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122）；3 视图集合 `[23,24,33]` 与 `[42,43,44]`，脚本支持扩展到 4/6 视图 | CD、Precision、Recall（1 mm 阈值） |
| **Tanks and Temples** | 官方下载页 <https://www.tanksandtemples.org/download/>；工具箱 [TanksAndTemples](https://github.com/isl-org/TanksAndTemples) | Intermediate 8 场景（Family / Francis / Horse / Lighthouse / M60 / Panther / Playground / Train）；Advanced 6 场景；Training 7 场景。测试集真值不公开，需在线提交评测 | F-Score（按场景使用官方 dTau） |

`scripts/download_datasets.py --print` 打印下载说明；`--verify <manifest.json>` 检查本地文件是否齐全。

---

## 3. 目录结构

```
.
├── README.md
├── requirements.txt
├── configs/default.json         模型与训练超参（w=8, s=4, AdamW 1e-4, cosine, 512×384…）
├── vggt_al/                     核心代码包
│   ├── attention.py             AGLA / AGLABlock / 窗口注意力
│   ├── model.py                 VGGT-AL 编码器-解码器、深度/相机头、点云反投影
│   ├── losses.py                多任务损失
│   ├── data.py                  统一 manifest 数据集接口
│   ├── metrics.py               CD / F-Score / NC / Precision / Recall
│   └── utils.py                 配置、断点、PLY 导出等
├── scripts/
│   ├── download_datasets.py     数据下载说明与本地校验
│   ├── prepare_ss3dm.py         SS3DM → manifest
│   ├── prepare_dtu.py           DTU → manifest（3/4/6 视图）
│   ├── prepare_tnt.py           Tanks and Temples → manifest
│   ├── train.py                 训练
│   ├── evaluate.py              推理 + 指标 + PLY 导出
│   └── smoke_test.py            端到端自检（无需下载数据）
└── tests/                       单元测试
```

`manifest` 是三个数据集共用的 JSON 索引，包含每个场景的图像路径、`3×3` 内参、`4×4` 相机外参（可选）、深度图（可选）、GT 点云路径（可选）。

---

## 4. 快速开始

```bash
# 环境
pip install -r requirements.txt

# 自检（不需要下载数据）
python scripts/smoke_test.py
python -m unittest discover -s tests -v

# 查看数据下载说明
python scripts/download_datasets.py --print

# 数据准备（路径换成你自己的）
python scripts/prepare_dtu.py --data_root /data/DTU --split train --num_views 6 \
    --out manifests/dtu_train_6v.json
python scripts/prepare_dtu.py --data_root /data/DTU --split test --num_views 6 \
    --out manifests/dtu_test_6v.json
python scripts/prepare_ss3dm.py --data_root /data/SS3DM --out manifests/ss3dm_6v.json
python scripts/prepare_tnt.py --data_root /data/TanksAndTemples \
    --group intermediate --out manifests/tnt_6v.json

# 训练 / 评估
python scripts/train.py --config configs/default.json \
    --manifest manifests/dtu_train_6v.json --out checkpoints/dtu_al
python scripts/evaluate.py --config configs/default.json \
    --manifest manifests/dtu_test_6v.json \
    --checkpoint checkpoints/dtu_al/epoch_200.pt --out outputs/dtu_al
```

`configs/default.json` 中 `epochs=200`、`lr=1e-4`、`weight_decay=1e-2`、cosine 调度与论文设置一致；默认 `batch_size=8` 面向单卡运行，多卡环境可调至论文的 32。`--max-steps 20` 可用于快速验证训练流程。

`VGGTAL.init_from_vggt(state_dict)` 可将 VGGT 预训练注意力权重载入全局分支；局部分支与门控按论文方式初始化，避免破坏预训练表征。

---

## 5. 实现说明

1. **模型规模**：默认配置为便于单卡运行的轻量设置（`embed_dim=384, depth=12`）；可通过 `configs/default.json` 调整层数与维度。
2. **SS3DM 划分**：`prepare_ss3dm.py` 提供确定性的场景级划分，并记录在 manifest 元数据（`split` 字段）中，便于按同一划分重复实验。
3. **Tanks and Temples 评测**：测试集激光真值不公开，本地推理只导出 PLY；正式 F-Score 请用官方评测服务。`eval.cd_threshold` 默认 0.05（SS3DM），DTU 请改为 0.001（1 mm）。
4. **CD 量纲**：`metrics.chamfer_distance` 返回点云原始坐标单位下的距离，跨数据集对比前请统一场景尺度。
5. **相机位姿监督**：manifest 提供 `poses` 时启用 `camera` 损失；无位姿标注的数据集自动关闭该项，相机几何由光度重投影损失监督。
6. **可选依赖**：无 `scipy` 时指标自动使用 NumPy 分块近邻搜索（结果一致，速度较慢）；无 `lpips` 时感知项使用可微的边缘先验，安装 `lpips` 并设置 `loss.use_lpips=true` 可切换为 LPIPS。

---

## 6. 引用

如使用本代码，请引用：

```
Adaptive Global-Local Attention for Large-Scale 3D Scene Reconstruction from Sparse Views
```

数据集请分别引用 SS3DM (NeurIPS 2024)、DTU MVS (CVPR 2014 / IJCV 2016) 与 Tanks and Temples (CVPR 2017)。
