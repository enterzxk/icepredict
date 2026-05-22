# Python 3.9 服务器部署指南

## 快速安装

### 1. 上传代码到服务器

PyCharm中：`Tools → Deployment → Upload to...`

### 2. SSH连接服务器

```bash
ssh user@server
cd /path/to/project
```

### 3. 激活虚拟环境

```bash
# 如果使用venv
source venv/bin/activate

# 或者如果虚拟环境在其他位置
source /path/to/venv/bin/activate
```

### 4. 一键安装

```bash
bash install_server_py39.sh
```

### 5. 设置API Key

```bash
export MIMO_API_KEY="your_mimo_api_key"

# 永久生效
echo 'export MIMO_API_KEY="your_mimo_api_key"' >> ~/.bashrc
source ~/.bashrc
```

### 6. 运行脚本

```bash
python run_auto_label.py      # 数据标注
python run_split_dataset.py   # 数据集划分
python run_train_server.py    # 训练模型
python run_evaluate.py        # 评估模型
python run_inference.py       # 推理测试
```

## 详细步骤

### 步骤1: 检查环境

```bash
# 检查Python版本
python3 --version  # 应显示 Python 3.9.25

# 检查pip
pip --version

# 检查虚拟环境
which python3
```

### 步骤2: 安装依赖

```bash
# 使用Python 3.9兼容版本
pip install -r requirements_py39.txt

# 如果需要GPU版本PyTorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 步骤3: 验证安装

```bash
python3 -c "
import torch
import anthropic
from tools.auto_label import MultiLabelAutoLabeler

print('所有依赖安装成功！')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### 步骤4: 数据标注

```bash
# 设置API Key
export MIMO_API_KEY="your_api_key"

# 运行标注
python run_auto_label.py
```

### 步骤5: 数据集划分

```bash
python run_split_dataset.py
```

### 步骤6: 训练模型

```bash
# 前台训练
python run_train_server.py

# 或后台训练（推荐）
nohup python run_train_server.py > train.log 2>&1 &

# 查看训练日志
tail -f train.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 步骤7: 评估模型

```bash
python run_evaluate.py
```

## 常见问题

### Q1: 依赖版本冲突

```bash
# 创建新的虚拟环境
python3 -m venv venv
source venv/bin/activate

# 重新安装
pip install -r requirements_py39.txt
```

### Q2: PyTorch没有GPU支持

```bash
# 卸载当前PyTorch
pip uninstall torch torchvision

# 安装GPU版本（CUDA 11.7）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 或CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Q3: anthropic库版本过低

```bash
pip install --upgrade anthropic
```

### Q4: 内存不足

编辑 `run_train_server.py`，减小batch_size：
```python
args = argparse.Namespace(
    batch_size=16,  # 或更小
    # ...
)
```

### Q5: 训练太慢

编辑 `run_train_server.py`，减少训练轮数：
```python
args = argparse.Namespace(
    epochs_stage1=5,
    epochs_stage2=20,
    # ...
)
```

## 后台训练管理

### 启动后台训练

```bash
nohup python run_train_server.py > train.log 2>&1 &
echo $! > train.pid
```

### 查看训练状态

```bash
# 查看日志
tail -f train.log

# 查看进程
ps aux | grep run_train_server

# 查看GPU使用
nvidia-smi
```

### 停止训练

```bash
kill $(cat train.pid)
```

## 下载结果

训练完成后，下载结果到本地：

### 使用PyCharm
```
Tools → Deployment → Download from...
```

### 使用scp
```bash
scp -r user@server:/path/to/project/weights/ice_classifier ./weights/
scp -r user@server:/path/to/project/experiments/eval_figures ./experiments/
```

## 完整流程示例

```bash
# 1. 连接服务器
ssh user@server

# 2. 进入项目
cd /path/to/project

# 3. 激活虚拟环境
source venv/bin/activate

# 4. 安装依赖
bash install_server_py39.sh

# 5. 设置API Key
export MIMO_API_KEY="your_key"

# 6. 数据标注
python run_auto_label.py

# 7. 数据集划分
python run_split_dataset.py

# 8. 后台训练
nohup python run_train_server.py > train.log 2>&1 &

# 9. 监控训练
tail -f train.log

# 10. 训练完成后评估
python run_evaluate.py

# 11. 下载结果到本地
```

## 依赖版本说明

Python 3.9 兼容版本：

| 包 | 版本范围 |
|---|---------|
| torch | >=1.12.0, <2.0.0 |
| torchvision | >=0.13.0, <0.15.0 |
| numpy | >=1.21.0, <1.24.0 |
| pandas | >=1.3.0, <2.0.0 |
| matplotlib | >=3.4.0, <3.7.0 |
| anthropic | >=0.18.0 |
| openai | >=1.0.0 |
