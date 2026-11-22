# VGGT-AL — 论文代码复现

本目录是根据论文 **《Adaptive Global-Local Attention for Large-Scale 3D Scene Reconstruction from Sparse Views》** 生成的独立代码仓库。核心贡献 **AGLA（Adaptive Global-Local Attention）** 以纯 PyTorch 实现，可直接替换 VGGT 编码器中的标准自注意力层；同时提供对接 **SS3DM / DTU / Tanks and Temples** 三个实验数据集的数据准备、训练、评估脚本。

> 说明：还没有执行任何 `git` 操作（没有 init / commit / push）。是否提交到 GitHub 由你检查后自行决定。

---

## 1. 论文方法概要

| 论文要点 | 本仓库实现 |
|---|---|
| 全局分支：对特征图做 `s=4` 平均池化后做标准 MHSA，再双线性上采样回原分辨率 | `vggt_al/attention.py` → `MultiHeadSelfAttention` + `AGLA.forward` 的 global 分支 |
| 局部分支：原分辨率、非重叠 `8×8` 窗口内自注意力 | `WindowSelfAttention`（含相对位置偏置，支持任意分辨率 padding） |
| 可学习门控：轻量卷积输出 2 通道，softmax 后逐元素融合两分支 | `AGLA.gate_conv`，门控图 `G ∈ R^(B,V,2,H,W)`，初始为 0.5/0.5 |
| 替换 VGGT 编码器注意力，保留位置编码、FFN、解码器 | `AGLABlock` + `VGGTAL`，帧内 / 跨视图注意力交替 |
| 多任务损失：深度回归、辐射场重建、相机位姿约束、融合权重平滑 | `vggt_al/losses.py` → `VGGTALLoss` |
| 输出深度图、置信度、相机位姿、点云 | `VGGTAL.forward` 返回 `depth / confidence / cameras / points` |

按论文文字描述（原 docx 中公式是 OMML 对象，抽取纯文本时只剩公式编号），AGLA 可写为：

```
Xg = Upsample( MHSA( LN( AvgPool2d(X, stride=s=4) ) ) )   # 跨视图或单视图，取决于 block 的 scope
Xl = WindowMHSA( LN(X), window=8 )                        # 逐视图
G  = softmax( Conv3x3(X), dim=channel )                   # 两个空间权重图，和为 1
Y  = Linear( G0 ⊙ Xg + G1 ⊙ Xl )
```

消融模式通过 `config.model.fusion_mode` 切换：`gated`（完整方法）、`concat`（固定拼接）、`global_only`、`local_only`，对应论文 Table 2。

---

## 2. 实验数据集（已查证官方来源）

| 数据集 | 官方来源与下载 | 规模与协议 | 论文中的指标 |
|---|---|---|---|
| **SS3DM**（Synthetic Street Scene 3D Mesh） | 项目主页 <https://ss3dm.top/>；导出器 [THU-LYJ-Lab/SS3DM-Exporter](https://github.com/THU-LYJ-Lab/SS3DM-Exporter)；评测与 GT 提取 [THU-LYJ-Lab/SS3DM-Benchmark](https://github.com/THU-LYJ-Lab/SS3DM-Benchmark)；数据由项目页的 Google Drive / Hugging Face 链接下载 | NeurIPS 2024，清华 + 地平线，基于 CARLA；6 路 RGB + 5 路 LiDAR，共 13,535 帧；14 short + 8 middle + 6 long 序列；提供精确合成网格。先用 `extract_gt_pcd.py` 生成参考点云 | CD、F-Score（t=0.05）、Normal Consistency |
| **DTU MVS** | 官方页 <https://roboimagedata.compute.dtu.dk/?page_id=36>；`SampleSet.zip`（6.3 GB，图像 + 标定）、`Points.zip`（6.3 GB，结构光参考点云） | 124 个场景，每场景 49 或 64 视角、7 种光照，1600×1200；原始论文评测 80 个场景；学习类方法常用 77 train / 15 test 划分，测试 ID 为 24,37,40,55,63,65,69,83,97,105,106,110,114,118,122；标准 3 视图集合 `[23,24,33]` 与 `[42,43,44]` | CD、Precision、Recall（1 mm 阈值） |
| **Tanks and Temples** | 官方下载页 <https://www.tanksandtemples.org/download/>；工具箱 [isl-org/TanksAndTemples](https://github.com/isl-org/TanksAndTemples) | Intermediate 8 个场景（Family / Francis / Horse / Lighthouse / M60 / Panther / Playground / Train）；Advanced 6 个；Training 7 个（有 GT）。测试集真值不公开，需在线提交评测。推荐内参先验 `fx=fy=0.7W, cx=W/2, cy=H/2` | F-Score（按场景使用官方 dTau） |

`scripts/download_datasets.py --print` 会打印上述完整下载说明；`--verify <manifest.json>` 可检查本地文件是否齐全。

---

## 3. 目录结构

```
.
├── README.md                    本文件
├── requirements.txt
├── configs/default.json         论文超参（w=8, s=4, AdamW 1e-4, cosine, 512×384…）
├── vggt_al/                     核心代码包
│   ├── attention.py             AGLA / AGLABlock / 窗口注意力（核心贡献）
│   ├── model.py                 VGGT-AL 编码器-解码器、深度/相机头、点云反投影
│   ├── losses.py                四项损失 + 权重
│   ├── data.py                  统一 manifest 数据集接口
│   ├── metrics.py               CD / F-Score / NC / P / R
│   └── utils.py                 配置、断点、PLY 导出等
├── scripts/
│   ├── download_datasets.py     官方下载指引与本地校验
│   ├── prepare_ss3dm.py         SS3DM → manifest
│   ├── prepare_dtu.py           DTU → manifest（3/4/6 视图）
│   ├── prepare_tnt.py           Tanks and Temples → manifest
│   ├── train.py                 训练
│   ├── evaluate.py              推理 + 指标 + PLY 导出
│   └── smoke_test.py            无需真实数据的端到端冒烟测试
└── tests/                       unittest 单元测试
```

`manifest` 是三个数据集共用的 JSON 索引，包含每个场景的图像路径、`3×3` 内参、`4×4` 相机外参（可选）、深度图（可选）、GT 点云路径（可选）。

---

## 4. 快速开始

```bash
# 1) 环境
pip install -r requirements.txt

# 2) 冒烟测试（不需要下载数据，几秒完成）
python scripts/smoke_test.py
python -m unittest discover -s tests -v

# 3) 查看数据集下载方式
python scripts/download_datasets.py --print

# 4) 数据准备（路径换成你自己的）
python scripts/prepare_dtu.py --data_root /data/DTU --split train --num_views 6 \
    --out manifests/dtu_train_6v.json
python scripts/prepare_dtu.py --data_root /data/DTU --split test --num_views 6 \
    --out manifests/dtu_test_6v.json
# SS3DM: 先运行官方 extract_gt_pcd.py，再运行：
python scripts/prepare_ss3dm.py --data_root /data/SS3DM --out manifests/ss3dm_6v.json
# Tanks and Temples:
python scripts/prepare_tnt.py --data_root /data/TanksAndTemples \
    --group intermediate --out manifests/tnt_6v.json

# 5) 训练 / 评估
python scripts/train.py --config configs/default.json \
    --manifest manifests/dtu_train_6v.json --out checkpoints/dtu_al
python scripts/evaluate.py --config configs/default.json \
    --manifest manifests/dtu_test_6v.json \
    --checkpoint checkpoints/dtu_al/epoch_200.pt --out outputs/dtu_al
```

`configs/default.json` 中的 `train.epochs=200`、`lr=1e-4`、`weight_decay=1e-2`、cosine 调度与论文一致；默认 `batch_size=8` 是为了单卡可运行，论文为 8×A100、batch 32。可用 `--max-steps 20` 做小步训练验证。

`VGGTAL.init_from_vggt(state_dict)` 可把官方 VGGT 的注意力权重拷入全局分支；局部分支与门控保持论文规定的初始化，避免破坏预训练权重。

---

## 5. 复现实验时的注意点（重要）

1. **公式细节**：原 docx 里的公式是 Word 数学对象，文本抽取后只剩 `(1)~(13)` 编号。本实现严格按 Section 2.3/2.4 的文字描述编写，若作者原始实现有未披露的细节（窗口移位、门控卷积归一化方式、损失权重的具体数值等），结果可能有差异。
2. **预训练权重**：论文依赖 VGGT 预训练权重，完整 VGGT 模型与权重未随本文发布；本仓库是可运行的研究级轻量实现（默认 `embed_dim=384, depth=12`），不是官方 VGGT 的完整复刻。论文 Table 1 的绝对数值（如 SS3DM CD=0.029）需要在完整模型、完整数据和作者划分下才能直接对比。
3. **SS3DM 划分**：论文写的是 800/200 场景划分，但官方公开数据并未附带该划分 ID。`prepare_ss3dm.py` 提供确定性划分并记录在 manifest 元数据中，复现论文数字前请向作者确认划分。
4. **Tanks and Temples 评测**：测试集激光真值不公开；本仓库本地只导出 PLY，正式 F-Score 请用官方评测服务。`eval.cd_threshold` 默认 0.05（SS3DM），DTU 请改成 0.001（1 mm）。
5. **CD 量纲**：`metrics.chamfer_distance` 返回点云原始坐标单位下的距离；论文表格中的数值很可能经过场景归一化，比较前请统一尺度。
6. **相机位姿监督**：只有 manifest 里提供 `poses` 时 `camera` 损失才生效（DTU 标定、SS3DM 导出位姿）；Tanks and Temples 无位姿标注时该项自动关闭，相机几何由光度重投影损失监督。
7. **依赖降级**：没有 `scipy` 时指标会用 NumPy 分块暴力近邻搜索（较慢但结果一致）；没有 `lpips` 时感知项退化为可微的边缘先验，安装 `lpips` 并设置 `loss.use_lpips=true` 可切换到论文使用的 LPIPS。

---

## 6. 引用

若使用本代码，请引用论文原文：

```
Adaptive Global-Local Attention for Large-Scale 3D Scene Reconstruction from Sparse Views
```

数据集请分别引用 SS3DM (NeurIPS 2024)、DTU MVS (CVPR 2014 / IJCV 2016) 与 Tanks and Temples (CVPR 2017)。
