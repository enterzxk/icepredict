# PyCharm 远程服务器配置指南

## 1. 服务器环境准备

### 1.1 安装依赖

SSH连接到服务器后执行：

```bash
# 进入项目目录
cd /path/to/project

# 安装依赖
pip install -r requirements.txt

# 如果需要GPU版本PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 1.2 检查GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
```

## 2. PyCharm配置

### 2.1 配置远程解释器

1. `File` → `Settings` → `Project` → `Python Interpreter`
2. 点击齿轮图标 → `Add` → `SSH Interpreter`
3. 输入服务器信息：
   - Host: 服务器IP
   - Port: 22
   - Username: 用户名
4. 选择解释器路径：`/usr/bin/python3` 或 `/path/to/venv/bin/python`
5. 配置路径映射：
   - Local: `C:\Users\京康\Desktop\2026023169-作品主文件夹\2026023169-02素材与源码`
   - Remote: `/path/to/project`

### 2.2 配置Deployment

1. `Tools` → `Deployment` → `Configuration`
2. 添加SFTP服务器
3. 配置映射（Mapping标签）：
   - Local path: 本地项目路径
   - Deployment path: `/path/to/project`
4. 右键项目 → `Deployment` → `Upload to...` 上传代码

## 3. 运行配置

### 3.1 创建运行配置

`Run` → `Edit Configurations` → `+` → `Python`

#### 配置1: 数据标注
```
Name: Auto Label
Script: run_auto_label.py
Working directory: /path/to/project
Environment variables: DASHSCOPE_API_KEY=your_api_key
```

#### 配置2: 数据集划分
```
Name: Split Dataset
Script: run_split_dataset.py
Working directory: /path/to/project
```

#### 配置3: 训练模型
```
Name: Train Classifier
Script: run_train.py
Working directory: /path/to/project
```

#### 配置4: 评估模型
```
Name: Evaluate Classifier
Script: run_evaluate.py
Working directory: /path/to/project
```

#### 配置5: 推理测试
```
Name: Inference Test
Script: run_inference.py
Working directory: /path/to/project
```

## 4. 服务器专用脚本

### 4.1 一键安装脚本

```bash
#!/bin/bash
# setup_server.sh

echo "=== 安装依赖 ==="
pip install -r requirements.txt

echo "=== 检查GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo "=== 创建目录 ==="
mkdir -p data/labels data/dataset weights/ice_classifier runs experiments/eval_figures

echo "=== 完成 ==="
```

### 4.2 训练脚本（服务器优化版）

```python
# run_train_server.py
# 服务器训练配置，使用更多GPU资源

args = argparse.Namespace(
    data_dir="data/dataset",
    batch_size=64,           # 服务器GPU内存大，可以用更大batch
    epochs_stage1=10,
    epochs_stage2=50,        # 服务器可以训练更多轮
    lr_stage1=1e-3,
    lr_stage2=1e-4,
    num_workers=8,           # 服务器CPU核心多
    device="cuda:0",         # 明确指定GPU
    # ...
)
```

## 5. 常见问题

### Q1: 代码同步问题

```bash
# 手动上传代码
scp -r ./* user@server:/path/to/project/

# 或使用rsync增量同步
rsync -avz --exclude='.idea' --exclude='__pycache__' ./ user@server:/path/to/project/
```

### Q2: 路径不一致

确保本地和服务器的路径映射正确：
- PyCharm会自动处理路径转换
- 在代码中使用相对路径

### Q3: GPU内存不足

```python
# 减小batch_size
args = argparse.Namespace(
    batch_size=16,  # 或更小
    # ...
)
```

### Q4: 训练中断恢复

```python
# 在run_train.py中设置resume
args = argparse.Namespace(
    resume="weights/ice_classifier/latest_stage2.pth",
    # ...
)
```

## 6. 完整流程

```bash
# 1. 上传代码到服务器
# PyCharm: Tools → Deployment → Upload to...

# 2. SSH连接服务器
ssh user@server

# 3. 进入项目目录
cd /path/to/project

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行数据标注（需要API Key）
export DASHSCOPE_API_KEY="your_key"
python run_auto_label.py

# 6. 划分数据集
python run_split_dataset.py

# 7. 训练模型
python run_train.py

# 8. 评估模型
python run_evaluate.py

# 9. 下载结果到本地
# PyCharm: Tools → Deployment → Download from...
```

## 7. 使用nohup后台训练

```bash
# 后台训练，即使断开SSH也不会中断
nohup python run_train.py > train.log 2>&1 &

# 查看训练日志
tail -f train.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```
