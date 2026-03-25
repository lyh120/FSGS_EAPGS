# FSGS_EAPGS - 低光照场景少样本视图合成

本项目是一个结合 **FSGS (Few-Shot Gaussian Splatting)** 和 **EAP-GS** 的低光照场景 3D 重建系统。通过 Retinexformer 预处理增强黑暗图像，结合双路径 3D 重建与后处理，实现高质量的少样本视图合成。

---

## 目录

- [项目概述](#项目概述)
- [技术架构](#技术架构)
- [Pipeline 流程](#pipeline-流程)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心模块说明](#核心模块说明)
- [依赖项目与许可](#依赖项目与许可)
- [许可证](#许可证)

---

## 项目概述

本项目针对 **低光照/暗光场景** 的少样本 (Few-Shot) 视图合成问题，提出了一套完整的后期融合 Pipeline：

1. 使用 **Retinexformer** (基于 LOL-v1 / LOL-v2_real 预训练模型) 对输入图像进行亮度增强预处理
2. 分别使用 **FSGS** 和 **EAP-GS** 两条独立路径进行 3D 重建
3. 渲染阶段应用 Gamma 校正等图像增强模块
4. 通过 **直方图匹配 (Histogram Matching)** 将渲染结果的像素分布与原图拉齐
5. 人工介入，筛选两条路径中的最优结果

---

## 技术架构

### 核心技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 图像增强 | Retinexformer | 低光照图像增强预训练模型 |
| 3D 重建 (路径一) | FSGS | Few-Shot Gaussian Splatting，支持稀疏视角重建 |
| 3D 重建 (路径二) | EAP-GS | 深度感知增强的 Gaussian Splatting |
| 渲染增强 | Gamma/Brightness/Contrast | 渲染后处理增亮增对比度 |
| 颜色校正 | Histogram Matching | 直方图匹配保持像素分布一致性 |

### 关键改进

#### FSGS 分支 (根目录)
- **随机初始化点云**：将原本依赖 COLMAP SfM 重建的稀疏点云，替换为随机初始化的点云 (`data_convert.py:155-161`)
- **调整训练参数**：修改了 `arguments/__init__.py` 中的优化参数，适应随机初始化场景
- **渲染增强**：在 `render.py` 中新增 `render_enhance()` 函数，对渲染结果进行 Gamma 校正、亮度/对比度/饱和度调整

#### EAP-GS 分支 (`EAP-GS/`)
- **Blender 到 COLMAP 格式转换**：通过 `blender_to_colmap.py` 将 Blender 格式数据 (`transforms_train.json`, `transforms_test.json`) 转换为 COLMAP 稀疏模型格式
- **COLMAP 重建流程**：使用 `run_colmap.py` 对 train_images 和 test_images 分别运行 COLMAP 重建
- **渲染增强**：同样包含 `render_enhance()` 后处理函数

---

## Pipeline 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入: 黑暗场景多视角图像                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Step 1: Retinexformer 预处理                    │
│  - 使用 LOL-v1 / LOL-v2_real 预训练模型                          │
│  - 将黑暗图像增强为明亮图像                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│      FSGS 重建路径          │   │      EAP-GS 重建路径      │
│  (随机初始化点云)           │   │  (COLMAP + ZoeDepth)      │
│                           │   │                           │
│  - data_convert.py        │   │  - blender_to_colmap.py   │
│  - train.py               │   │  - run_colmap.py          │
│  - render.py              │   │  - train.py               │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Step 3-4: 渲染 + 图像增强                        │
│  - render_enhance(): Gamma(0.82), Brightness(0.12),             │
│                      Contrast(1.4), Saturation(1.40)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Step 5: 直方图匹配 (Histogram Matching)        │
│  - histogram_match/main.py                                      │
│  - 计算原图亮度统计 (mean/std)                                    │
│  - 将渲染图亮度分布拉到与原图一致                                  │
│  - 保护高光/暗部区域避免过曝或死黑                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Step 6: 人工筛选最优结果                        │
│  - 对比 FSGS 和 EAP-GS 两条路径的结果                              │
│  - 人工选取质量更高的渲染图                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         最终输出: 高质量新视角图像                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
FSGS_EAPGS/
│
├── README.md                    # 本项目说明文档
├── LICENSE.md                   # 许可证文件 (Gaussian-Splatting License)
│
├── FSGS 分支 (主目录)
│   ├── train.py                 # FSGS 训练脚本
│   ├── render.py                # FSGS 渲染脚本 (含 render_enhance)
│   ├── metrics.py               # 评估指标计算
│   ├── full_eval.py             # 完整评估流程
│   ├── data_convert.py          # Blender → COLMAP 格式转换 (随机点云)
│   ├── blender_to_colmap.py     # Blender JSON 转 COLMAP 稀疏模型
│   ├── convert.py               # COLMAP SfM 流程脚本
│   ├── arguments/              # 训练参数定义
│   ├── scene/                   # 场景/相机/高斯模型
│   ├── gaussian_renderer/       # 可微高斯渲染器
│   ├── utils/                   # 工具函数
│   ├── tools/                   # COLMAP 预处理工具
│   ├── lpipsPyTorch/            # LPIPS 感知损失
│   └── submodules/              # 子模块 (simple-knn, diff-gaussian-rasterization)
│
├── EAP-GS/                      # EAP-GS 分支
│   ├── train.py                 # EAP-GS 训练脚本
│   ├── run_colmap.py            # COLMAP SfM 流程
│   ├── blender_to_colmap.py     # Blender 转 COLMAP
│   ├── augmentation.py          # 数据增强
│   ├── ZoeDepth/                # 深度估计模块
│   ├── scene/                   # 场景/相机/高斯模型
│   ├── gaussian_renderer/       # 可微高斯渲染器
│   ├── utils/                   # 工具函数
│   ├── arguments/               # 训练参数定义
│   ├── lpipsPyTorch/            # LPIPS 感知损失
│   ├── dataset_eap/             # EAP 数据集处理
│   └── submodules/              # 子模块
│
├── histogram_match/             # 直方图匹配模块
│   ├── main.py                  # 主脚本 (亮度统计 + 直方图匹配)
│   └── main_color.py            # 颜色直方图匹配变体
│
├── dataset_all/                  # 原始数据集
│   └── dataset_v2/              # 数据集 v2 格式
│       └── {SceneName}/
│           ├── train/           # 训练图像
│           ├── transforms_train.json
│           └── transforms_test.json
│
├── colmap_unity/                # Unity 采集数据的 COLMAP 结果
├── colmap/                      # 其他 COLMAP 数据
├── output_all/                  # FSGS 输出结果
├── logs_*.txt                   # 训练日志
│
└── environment.yml              # Conda 环境配置
```

---

## 快速开始

### 环境安装

```bash
# 创建 Conda 环境
conda env create --file environment.yml
conda activate FSGS

# 安装子模块 
# submodules/simple-knn
# submodules/diff-gaussian-rasterization
```

> **注意**: CUDA 11.7 是推荐的版本。相关具体环境安装指南请参考项目基于的其他项目指南。

### 数据准备

#### 方式一：使用已有 Blender 格式数据

```bash
# 数据集格式
dataset_v2/
└── SceneName/
    ├── train/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── transforms_train.json    # 训练集相机参数
    └── transforms_test.json     # 测试集相机参数
```

#### 方式二：运行 Retinexformer 预处理

```bash
# 使用 LOL-v2_real 或 LOL-v1 预训练模型处理黑暗图像
# 参考 Retinexformer 官方仓库进行图像增强
```

### FSGS 训练

```bash
# 随机初始化点云 + 训练
# 训练
python train.py \
    --source_path dataset_v2/SceneName \
    --model_path output_v2/SceneName \
    --n_views 0 \
    --eval \
    --resolution 1 \
    --iterations 30000 \
    --test_iterations 1000 2000 5000 10000 20000 30000 \
    --save_iterations 10000 20000 30000 \
    --checkpoint_iterations 10000 20000 30000
# 渲染
python render.py --model_path output_v2/SceneName --iteration 60000 --resolution 1
# 结果
dataset_v2包含了训练数据，output_v2包含了训练好的模型权重。
```

### EAP-GS 训练

```bash
# Step 1: Blender → COLMAP 格式转换以及COLMAP 重建，批量运行脚本已经写好在start.sh中了，可以查看bash指令运行。
python blender_to_colmap.py \
    --train-json dataset_v2/SceneName/transforms_train.json \
    --test-json dataset_v2/SceneName/transforms_test.json \
    --image-root dataset_v2/SceneName \
    --existing-colmap colmap_unity/SceneName/sparse/0/train \
    --output-dir EAP-GS/dataset_eap/SceneName/sparse/0 \
    --split-subdirs

# Step 2: SfM
# Step 2.1: 使用COLMAP 重建 (可选，使用已有结果可跳过)
cd EAP-GS
python run_colmap.py --source_path dataset_eap/SceneName --camera PINHOLE

# Step 2.2: 使用VGGT 重建 (可选，需安装vggt)
python convert/vggt_nocol/depth.py
python convert/vggt_nocol/voxelize.py

# Step 3: 训练
python train.py \
    --source_path dataset_eap/SceneName \
    --model_path output_eap/SceneName \
    --eval \
    --resolution 1 \
    --iterations 30000
```

### 渲染

```bash
# FSGS 渲染
python render.py \
    --source_path dataset_colmap/SceneName \
    --model_path output/SceneName \
    --iteration 30000
```

### 后处理

提供两种后处理方法，

**方法1**
依据validation中BlueHawaii场景，对渲染结果的亮度进行调整
```bash
cd histogram_match

# 准备原图目录 (real_image/) 和渲染结果目录 (final/)
# 运行直方图匹配
python main.py
python main_color.py

# 参数调整 (在 main.py 中修改)
MATCH_STRENGTH = 1.0  # 1.0 = 完全套用目标亮度统计
```

**方法2**
参照渲染结果中色卡颜色，调整亮度、gamma、对比度等参数，这部分以及通过函数形式写入渲染过程中了，默认启用。
```bash
python /convert/relight_yrestrict.py
```


---

## 核心模块说明

### 1. `render_enhance()` - 渲染增强

位于 `render.py:33-69` 和 `EAP-GS/train.py:39-75`

```python
def render_enhance(image,
    brightness=0.12,   # 亮度偏移
    gamma=0.82,        # Gamma < 1 提亮
    contrast=1.4,     # 对比度
    saturation=1.40): # 饱和度
```

### 2. `data_convert.py` - 随机点云生成

将 Blender JSON 转换为 COLMAP 格式，但使用随机点云代替 SfM 重建结果：

```python
# 生成 100,000 个随机点
for pid in range(1, 100000):
    xyz = np.random.uniform(-3, 3, 3).tolist()
    points.append((pid, xyz, [128,128,128], 1.0, []))
```

### 3. 直方图匹配 (`histogram_match/main.py`)

- 计算原图亮度均值和标准差
- 对渲染图进行 Z-score 归一化到目标分布
- 保护高光 (>220) 和暗部 (<30) 区域避免失真

### 4. `blender_to_colmap.py` - Blender 格式转换

将 NeRF/Blender 格式的 `transforms.json` 转换为 COLMAP 稀疏模型 (.bin)，支持：
- 复用已有 COLMAP 模型的 train 观测
- 生成 train/test 子集
- 坐标系统转换 (OpenGL → OpenCV)

---

## 依赖项目与许可

本项目基于以下开源项目构建，受其各自许可证约束：

| 项目 | 许可证 | 用途 |
|------|--------|------|
| [FSGS](https://github.com/VITA-Group/FSGS) | Gaussian-Splatting License | Few-Shot Gaussian Splatting 核心 |
| [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Gaussian-Splatting License | 3DGS 基础框架 |
| [EAP-GS](https://github.com/your-eapgs-repo) | (请参考原项目) | 深度感知 Gaussian Splatting |
| [RetinexFormer](https://github.com/Guo砚砚/RetinexFormer) | (请参考原项目) | 低光照图像增强 |
| [ZoeDepth](https://github.com/isl-org/ZoeDepth) | (请参考原项目) | 单目深度估计 |
| [VGGT](https://github.com/facebookresearch/vggt) | (请参考原项目) | 深度估计、稀疏点云生成 |

### 子模块

- `submodules/simple-knn` - KNN 算子
- `submodules/diff-gaussian-rasterization` - 可微渲染器

---

## 许可证

本项目整体基于 **Gaussian-Splatting License** 发布，详情请参阅 `LICENSE.md`。

**重要条款摘要：**

- **允许**: 非商业性研究和评估使用
- **允许**: 复制、准备衍生作品、公开展示和分发
- **禁止**: 未经授权的商业使用
- **要求**: 衍生作品需包含本许可证副本
- **要求**: 发表/出版时需引用 Gaussian-Splatting 相关论文

**引用要求：**

如果您在研究中使用了本项目代码或受其启发，请同时引用：

```
# Gaussian-Splatting
@article{kerbl2023gaussian,
  title={Compact 3D Scene Representation via Sparse Gaussian Splatting},
  author={Kerbl, Bernhard and Kopanas, Georgios and Drettakis, George},
  year={2023}
}

# FSGS
@misc{zhu2023FSGS,
  title={FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting},
  author={Zhu, Zehao and Fan, Zhiwen and Jiang, Yifan and Wang, Zhangyang},
  year={2023},
  eprint={2312.00451},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

---

## 致谢

特别感谢以下优秀开源项目：

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [EAP-GS](https://github.com/your-eapgs-repo)
- [DreamGaussian](https://github.com/ashawkey/diff-gaussian-rasterization)
- [RetinexFormer](https://github.com/Guo砚砚/RetinexFormer)
- [ZoeDepth](https://github.com/isl-org/ZoeDepth)
- [VGGT](https://github.com/facebookresearch/vggt)
