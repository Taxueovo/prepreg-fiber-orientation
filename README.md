# DINOv3–ConvNeXt Fibre Orientation Regression

一个用于碳纤维预浸料表面纤维锐角取向回归的研究代码库。模型以 ConvNeXt-Tiny 为空间骨干，可加载兼容的 DINOv3 预训练权重，并组合方向选择注意力、FFT 环带特征和异方差不确定性预测。

本仓库是去敏后的代码发布版：不包含原始图像、标签清单、模型权重、训练结果、论文草稿、作者信息或本地路径。它也不声明任何未随仓库公开验证的性能数字。

## 功能

- 母图隔离的数据划分接口，训练/验证/测试各自使用 CSV 与图像目录。
- ConvNeXt-Tiny 空间骨干与可选方向注意力模块。
- 可选 FFT 环带频域分支。
- 锐角回归与可选异方差损失。
- 验证集早停、逐轮指标、最佳模型与相对路径预测明细。
- GST（Gradient Structure Tensor）传统视觉基线。
- 预训练对照、析因消融、外部域、不确定性、计量、标签效率、旋转等变性和延迟分析脚本。

## 环境

建议使用 Python 3.10 或 3.11。根据你的 CUDA/ROCm/CPU 环境先安装匹配版本的 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据格式

数据由使用者自行准备，不提交到 Git。默认目录结构如下：

```text
database/
├── train/
│   ├── train.csv
│   └── images/
├── val/
│   ├── val.csv
│   └── images/
└── test/
    ├── test.csv
    └── images/
```

每个 CSV 至少需要以下字段：

```csv
patch_filename,source_image,angle,x,y
sample_0001.jpg,source_001.jpg,45.0,0,0
```

`patch_filename` 必须是相对于对应 `images/` 目录的安全相对路径，`angle` 会折叠到 `[0°, 90°]`。`source_image`、`x`、`y` 便于追踪母图与裁块，但当前训练加载器只直接读取 `patch_filename` 和 `angle`。

## 训练

```bash
python DinoConv.py \
  --data-dir /path/to/database \
  --weights /path/to/dinov3_convnext_tiny.pth \
  --require-pretrained \
  --device cuda:0 \
  --epochs 150 \
  --batch-size 64
```

不传 `--weights` 时模型使用随机初始化；正式的预训练对照建议同时使用 `--require-pretrained`，避免检查点缺失时静默回退。使用 `--device auto` 可自动选择 CUDA 或 CPU，也可显式指定 `cpu`、`cuda:0` 或 `mps`。

以下环境变量可代替常用路径参数：

```bash
export DINOCONV_DATA_DIR=/path/to/database
export DINOCONV_WEIGHTS=/path/to/checkpoint.pth
export DINOCONV_RUNS_DIR=/path/to/runs
export DINOCONV_DEVICE=cuda:0
```

训练产物默认写入 `runs/`，该目录已被 Git 忽略。配置快照和预测明细会移除工作站绝对路径。

## GST 基线

```bash
python gst_baseline.py \
  --csv /path/to/test.csv \
  --image_root /path/to/test/images \
  --output /path/to/gst_predictions.csv
```

## 补充实验

[thesis_experiment_packages/README.md](thesis_experiment_packages/README.md) 说明了八组实验脚本。训练脚本默认从 `DINOCONV_DATA_DIR` 或项目下的 `database/` 读取数据；所有生成结果均写入已忽略的 `thesis_experiment_packages/results/`。

## 检查

无需安装深度学习依赖即可运行发布层的轻量检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/dinoconv-pycache python -m compileall -q \
  DinoConv.py gst_baseline.py thesis_experiment_packages tests
python -m unittest discover -s tests -v
```

完整训练需要使用者提供合法获得的数据和兼容权重。请遵守数据、预训练模型及第三方依赖各自的许可条款。

## 许可

当前发布未附加开源许可证；默认保留全部权利。若计划允许他人复制、修改或再分发，请在明确授权范围后添加合适的许可证。
