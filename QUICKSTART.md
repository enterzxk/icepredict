# 快速开始 - PyCharm远程服务器

## 步骤1: 上传代码到服务器

在PyCharm中：
1. `Tools` → `Deployment` → `Upload to...`
2. 或右键项目 → `Deployment` → `Upload to...`

## 步骤2: SSH连接服务器

在PyCharm的Terminal中（确保已切换到远程终端）：

```bash
# 进入项目目录
cd /path/to/project

# 运行安装脚本
bash setup_server.sh
```

## 步骤3: 配置API Key

### 使用小米MiMo API（推荐）

```bash
export MIMO_API_KEY="your_mimo_api_key"
```

### 使用阿里云DashScope API

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

## 步骤4: 运行脚本

### 方式一: 在PyCharm中直接运行

在PyCharm中打开以下文件，点击运行按钮：

| 脚本 | 功能 |
|------|------|
| `run_auto_label_mimo.py` | 数据标注（小米MiMo版本） |
| `run_split_dataset.py` | 数据集划分 |
| `run_train_server.py` | 训练模型（服务器优化版） |
| `run_evaluate.py` | 评估模型 |
| `run_inference.py` | 推理测试 |

### 方式二: 在Terminal中运行

```bash
# 设置API Key（数据标注需要）
export MIMO_API_KEY="your_mimo_api_key"

# 1. 数据标注（小米MiMo版本）
python run_auto_label_mimo.py

# 2. 数据集划分
python run_split_dataset.py

# 3. 训练模型
python run_train_server.py

# 4. 评估模型
python run_evaluate.py

# 5. 推理测试
python run_inference.py
```

## 步骤4: 后台训练（推荐）

使用nohup后台训练，即使关闭PyCharm也不会中断：

```bash
nohup python run_train_server.py > train.log 2>&1 &
```

查看训练日志：
```bash
tail -f train.log
```

查看GPU使用：
```bash
watch -n 1 nvidia-smi
```

## 步骤5: 下载结果

训练完成后，在PyCharm中：
1. `Tools` → `Deployment` → `Download from...`
2. 下载 `weights/ice_classifier/` 目录
3. 下载 `experiments/eval_figures/` 目录

## 修改参数

如果需要调整训练参数，直接编辑 `run_train_server.py` 文件顶部的配置：

```python
args = argparse.Namespace(
    batch_size=64,           # 根据GPU内存调整
    epochs_stage2=50,        # 训练轮数
    lr_stage2=1e-4,          # 学习率
    # ...
)
```

## API配置

### 小米MiMo API

在 `config.py` 中修改：

```python
# 小米MiMo API配置
MIMO_CONFIG = {
    "base_url": "https://api.mimodel.com/v1",  # 小米MiMo API地址
    "model": "mimo-vl-plus",  # 模型名称
    "api_key_env": "MIMO_API_KEY",  # 环境变量名
}
```

### 阿里云DashScope API

```python
# 阿里云DashScope API配置
DASHSCOPE_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-vl-plus",
    "api_key_env": "DASHSCOPE_API_KEY",
}
```

## 常见问题

### Q: 内存不足
减小 `batch_size`：
```python
args = argparse.Namespace(
    batch_size=16,  # 或更小
    # ...
)
```

### Q: 训练太慢
减少训练轮数：
```python
args = argparse.Namespace(
    epochs_stage1=5,
    epochs_stage2=20,
    # ...
)
```

### Q: 权重下载失败
手动下载ResNet50权重并放到服务器：
```bash
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth -P ~/.cache/torch/hub/checkpoints/
```

### Q: API调用失败
检查API Key是否正确设置：
```bash
echo $MIMO_API_KEY
```

如果使用小米MiMo API，确保：
1. API Key正确
2. 网络可以访问小米API服务器
3. 模型名称正确（默认：mimo-vl-plus）
