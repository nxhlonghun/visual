# YOLOv8 训练工作区

这个仓库是一个基于 `Windows` 的 `YOLOv8` 本地训练与源码修改工作区。

## 环境信息

- 操作系统：`Windows`
- Python：`3.11.9`
- 虚拟环境：`.venv`
- PyTorch：`2.11.0+cu130`
- CUDA：`13.0`
- 显卡：`NVIDIA GeForce RTX 5060 Laptop GPU`

## 项目结构

```text
school/
  ultralytics-src/     # YOLOv8 源码目录，可直接修改
  README.md
  .gitignore
```

## 环境启动

在 PowerShell 中激活虚拟环境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "D:\task\school\.venv\Scripts\Activate.ps1"
```

检查当前环境是否正常：

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
yolo checks
```

## 完整环境配置流程

### 1. 安装 Python

建议安装 `Python 3.11.x`，本项目当前使用的是：

```text
Python 3.11.9
```

安装时建议勾选：

- `Add Python to PATH`

安装完成后检查：

```powershell
py -0p
py -3.11 --version
```

### 2. 创建项目目录

本项目根目录示例：

```text
D:\task\school
```

### 3. 创建虚拟环境

在项目根目录执行：

```powershell
py -3.11 -m venv .venv
```

如果 PowerShell 默认禁止执行脚本，可以先临时放开当前终端权限：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

然后激活虚拟环境：

```powershell
& "D:\task\school\.venv\Scripts\Activate.ps1"
```

激活成功后，终端前面通常会出现：

```text
(.venv)
```

### 4. 升级 pip

```powershell
python -m pip install --upgrade pip
```

### 5. 安装支持 RTX 5060 Laptop GPU 的 PyTorch

本项目显卡为：

```text
NVIDIA GeForce RTX 5060 Laptop GPU
```

经过测试，旧版 `cu121`、`cu126` 不适合当前显卡，最终可正常使用的版本是：

```text
torch 2.11.0+cu130
torchvision 0.26.0+cu130
CUDA 13.0
```

如果使用的是50系以下的gpu，则将CUDA 13.0更换为CUDA 12.1即可使用

安装命令如下：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

安装完成后验证 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

本项目期望看到的关键结果：

- `torch.__version__` 包含 `+cu130`
- `torch.version.cuda` 为 `13.0`
- `torch.cuda.is_available()` 为 `True`

### 6. 安装 YOLOv8

```powershell
python -m pip install ultralytics
```

验证安装：

```powershell
yolo checks
```

### 7. 切换为源码开发模式

如果只是训练模型，到上一步已经够用。  
如果需要长期修改 YOLO 源码，建议不要直接改 `.venv\Lib\site-packages\ultralytics`，而是拉一份源码单独开发。

克隆源码：

```powershell
git clone https://github.com/ultralytics/ultralytics.git D:\task\school\ultralytics-src
```

卸载普通安装版并切换到可编辑安装：

```powershell
python -m pip uninstall -y ultralytics
cd D:\task\school\ultralytics-src
python -m pip install -e .
```

检查当前是否已经使用源码版：

```powershell
python -c "import ultralytics; print(ultralytics.__file__)"
```

如果输出类似：

```text
D:\task\school\ultralytics-src\ultralytics\__init__.py
```

说明当前环境已经直接使用本地源码目录。

### 8. 最终环境检查

完成全部配置后，可以用以下命令统一检查：

```powershell
python --version
python -c "import ultralytics; print(ultralytics.__file__)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
yolo checks
```

如果输出满足以下条件，说明环境已经配置完成：

- Python 为 `3.11.x`
- `ultralytics` 路径正确
- `torch` 为 `+cu130`
- `torch.cuda.is_available()` 为 `True`
- `yolo checks` 中能看到 `CUDA:0`

## 源码开发模式

当前工作区使用的是 `ultralytics` 的可编辑安装模式，因此修改 `ultralytics-src` 中的代码后会直接生效。

检查当前实际加载的是哪一份源码：

```powershell
python -c "import ultralytics; print(ultralytics.__file__)"
```

如果需要重新安装为可编辑模式，请在源码根目录执行：

```powershell
cd D:\task\school\ultralytics-src
python -m pip install -e .
```

## 验证安装

```powershell
yolo detect predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg" device=0
```

## 模型训练

先用 1 轮快速测试训练流程是否正常：

```powershell
yolo detect train data="D:/task/school/dataset/data.yaml" model=yolov8n.pt epochs=1 imgsz=640 device=0
```

正式训练命令示例：

```powershell
yolo detect train data="D:/task/school/dataset/data.yaml" model=yolov8n.pt epochs=100 imgsz=640 device=0
```

常用训练参数（可按需叠加）：

- `data`：数据集配置文件路径（`data.yaml`）。
- `model`：预训练模型或已有权重（如 `yolov8n.pt`、`runs/detect/train/weights/last.pt`）。
- `epochs`：训练总轮数（例如 `100`、`200`）。
- `imgsz`：输入分辨率（常用 `640`；更大通常更准但更吃显存）。
- `batch`：每批图片数（越大训练越快但显存占用越高）。
- `device`：训练设备（`0` 表示 GPU0，`1` 表示 GPU1，`0,1` 表示多卡）。
- `workers`：数据加载线程数（Windows 建议从 `4` 或 `8` 开始调）。
- `save_period`：每隔多少轮额外保存一次 checkpoint（如 `10`）。
- `project`：结果输出根目录（默认 `runs/detect`）。
- `name`：本次训练子目录名（如 `train_coco_only`）。
- `exist_ok`：目录已存在时是否覆盖写入（`True/False`）。
- `patience`：早停耐心轮数，验证指标长期无提升时提前停止。
- `seed`：随机种子，用于增强复现实验结果。
- `resume`：断点续训开关（需要可恢复的 checkpoint）。
- `optimizer`：优化器策略（`auto`、`SGD`、`AdamW` 等）。
- `lr0`：初始学习率（`optimizer=auto` 时通常由框架自动调整）。
- `cos_lr`：是否使用余弦学习率调度（`True/False`）。
- `amp`：是否开启混合精度（一般保持 `True` 以降低显存占用）。

常见训练命令示例：

1) 固定每 10 轮保存一次，指定训练目录名：

```powershell
yolo detect train data="D:/task/school/dataset/data.yaml" model=yolov8n.pt epochs=200 imgsz=640 batch=16 device=1 save_period=10 project="D:/task/school/runs/detect" name="train200_mix"
```

2) 显存紧张时的保守配置：

```powershell
yolo detect train data="D:/task/school/dataset/data.yaml" model=yolov8n.pt epochs=100 imgsz=512 batch=8 device=0 workers=4
```

3) 断点续训到更高总轮数（目标总轮数写在 `epochs`）：

```powershell
yolo detect train resume model="D:/task/school/runs/detect/train5/weights/epoch20.pt" data="D:/task/school/dataset/data.yaml" epochs=50 device=0
```

提示：

- `epochs` 表示目标总轮数，不是“在当前基础上再加多少轮”。
- 想减少漏检可适当增加 `imgsz` 或降低推理阈值；想减少误检可提高推理阈值。
- 若命令很长，PowerShell 多行续行请在每行末尾加反引号 `` ` ``。

## 模型推理

使用训练完成后的模型进行预测：

```powershell
yolo detect predict model="D:/task/school/runs/detect/train/weights/best.pt" source="D:/task/school/test.jpg" device=0
```

推理/验证阈值常用参数：

- `conf`：置信度阈值（默认常用 `0.25`），越高误检越少、漏检可能越多。
- `iou`：NMS 阈值（默认常用 `0.7`），用于控制重叠框抑制强度。
- 说明：训练过程中每轮会执行 `val`，此时也会用到后处理（含 NMS）；你可以在训练命令中传 `conf/iou` 影响评估展示口径，但它不直接参与反向传播更新权重。
- 建议：先按常规参数完成训练，再在训练后通过 `yolo detect val/predict` 单独调 `conf/iou`，用于选择最终部署阈值。

阈值调节示例：

```powershell
yolo detect predict model="D:/task/school/runs/detect/train/weights/best.pt" source="D:/task/school/test.jpg" conf=0.25 iou=0.7 device=0
yolo detect val model="D:/task/school/runs/detect/train/weights/best.pt" data="D:/task/school/dataset/data.yaml" conf=0.25 iou=0.7 device=0
```

调参经验：

- 漏检偏多：适当降低 `conf`（如 `0.25 -> 0.15`）。
- 误检偏多：适当提高 `conf`（如 `0.25 -> 0.40`）。

## 数据集构建脚本

`build_dataset_from_sources.py` 支持按来源选择处理数据：

- `--source both`：同时处理 COCO + VisDrone（默认）
- `--source coco`：只处理 COCO
- `--source visdrone`：只处理 VisDrone

示例：

```powershell
python .\build_dataset_from_sources.py --source both
python .\build_dataset_from_sources.py --source coco
python .\build_dataset_from_sources.py --source visdrone
```

处理规则说明：

- 每次运行会先清理并重建合并后的 `images/`、`labels/` 结果目录（不是增量叠加）。
- `--source coco` 时，`train/val/test` 仅包含 COCO 样本。
- `--source visdrone` 时，`train/val/test` 仅包含 VisDrone 样本。
- `--source both` 时，`train/val/test` 同时包含两者样本。
- 默认会构建 `test`；如不需要可加 `--skip-test`。

如修改了源数据目录名（含内部子目录名），请在脚本顶部常量区调整对应配置。

## 说明

- `runs/` 目录用于保存训练和预测结果，已在 Git 中忽略。
- `*.pt` 等大模型文件已在 Git 中忽略。
- 如果需要修改 YOLO 源码，请编辑 `ultralytics-src/ultralytics` 目录中的文件。
