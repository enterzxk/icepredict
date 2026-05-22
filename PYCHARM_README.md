# PyCharm 运行指南

## 快速开始

### 1. 安装依赖

在PyCharm的Terminal中运行：

```powershell
python -m pip install -r requirements.txt
```

### 2. 运行脚本

所有脚本都已配置为**直接运行**，无需命令行参数。

#### 步骤1: 数据标注（需要API Key）

直接运行 `run_auto_label.py`

> 注意：需要设置环境变量 `DASHSCOPE_API_KEY`

#### 步骤2: 数据集划分

直接运行 `run_split_dataset.py`

#### 步骤3: 训练模型

直接运行 `run_train.py`

> 首次运行会自动下载预训练权重（约100MB）

#### 步骤4: 评估模型

直接运行 `run_evaluate.py`

#### 步骤5: 推理测试

直接运行 `run_inference.py`

## 脚本说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| `run_auto_label.py` | VLM自动标注 | `multi_label_results.json`, `data/labels/` |
| `run_split_dataset.py` | 数据集划分 | `data/dataset/` |
| `run_train.py` | 训练模型 | `weights/ice_classifier/` |
| `run_evaluate.py` | 评估模型 | `experiments/eval_figures/` |
| `run_inference.py` | 推理测试 | 控制台输出 |

## 修改参数

所有脚本的参数都在文件顶部的配置区域，可以直接修改：

```python
# 配置参数（可在PyCharm中直接修改）
args = argparse.Namespace(
    data_dir="data/dataset",        # 数据目录
    batch_size=32,                  # 批次大小
    epochs_stage1=10,               # 第一阶段轮数
    epochs_stage2=30,               # 第二阶段轮数
    use_focal_loss=True,            # 使用Focal Loss
    # ...
)
```

## GPU配置

脚本会自动检测GPU：
- 有CUDA GPU：自动使用 `cuda:0`
- 无GPU：使用 `cpu`

如需强制使用CPU，在 `run_train.py` 中修改：

```python
args = argparse.Namespace(
    # ...
    device="cpu",  # 强制使用CPU
    # ...
)
```

## 常见问题

### Q1: 缺少依赖包

```powershell
python -m pip install torch torchvision numpy pandas matplotlib scikit-learn tensorboard Pillow
```

### Q2: 权重下载失败

从以下地址手动下载ResNet50权重：
- https://download.pytorch.org/models/resnet50-11ad3fa6.pth

放到 `~/.cache/torch/hub/checkpoints/` 目录

### Q3: 内存不足

减小批次大小：

```python
args = argparse.Namespace(
    batch_size=16,  # 或 8
    # ...
)
```

### Q4: 训练太慢

减少训练轮数：

```python
args = argparse.Namespace(
    epochs_stage1=5,
    epochs_stage2=15,
    # ...
)
```

## 完整流程示例

```powershell
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 运行数据标注（需要API Key）
python run_auto_label.py

# 3. 划分数据集
python run_split_dataset.py

# 4. 训练模型
python run_train.py

# 5. 评估模型
python run_evaluate.py

# 6. 测试推理
python run_inference.py
```
