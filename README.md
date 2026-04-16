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

## 模型训练

先用 1 轮快速测试训练流程是否正常：

```powershell
yolo detect train data="D:/task/school/dataset/data.yaml" model=yolov8n.pt epochs=1 imgsz=640 device=0
```

正式训练命令示例：

```powershell
yolo detect train data="D:/task/school/dataset/data.yaml" model=yolov8n.pt epochs=100 imgsz=640 device=0
```

## 模型推理

使用训练完成后的模型进行预测：

```powershell
yolo detect predict model="D:/task/school/runs/detect/train/weights/best.pt" source="D:/task/school/test.jpg" device=0
```

## 说明

- `runs/` 目录用于保存训练和预测结果，已在 Git 中忽略。
- `*.pt` 等大模型文件已在 Git 中忽略。
- 如果需要修改 YOLO 源码，请编辑 `ultralytics-src/ultralytics` 目录中的文件。
