# 服务器部署指南

## 快速安装

### 1. 上传代码到服务器

使用PyCharm：
```
Tools → Deployment → Upload to...
```

或使用scp命令：
```bash
scp -r ./* user@server:/path/to/project/
```

### 2. SSH连接服务器

```bash
ssh user@server
cd /path/to/project
```

### 3. 运行安装脚本

```bash
bash install_server.sh
```

### 4. 设置API Key

```bash
# 小米MiMo API
export MIMO_API_KEY="your_mimo_api_key"

# 或者添加到 ~/.bashrc 永久生效
echo 'export MIMO_API_KEY="your_mimo_api_key"' >> ~/.bashrc
source ~/.bashrc
```

## 详细步骤

### 步骤1: 安装依赖

```bash
# 进入项目目录
cd /path/to/project

# 安装依赖
pip install -r requirements.txt

# 如果需要GPU版本PyTorch（如果pip安装的没有GPU支持）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步骤2: 检查环境

```bash
# 检查Python
python3 --version

# 检查PyTorch和GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查anthropic库
python3 -c "import anthropic; print(f'anthropic: {anthropic.__version__}')"
```

### 步骤3: 数据标注

```bash
# 设置API Key
export MIMO_API_KEY="your_api_key"

# 运行标注
python run_auto_label.py
```

### 步骤4: 数据集划分

```bash
python run_split_dataset.py
```

### 步骤5: 训练模型

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

### 步骤6: 评估模型

```bash
python run_evaluate.py
```

### 步骤7: 推理测试

```bash
python run_inference.py
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

## 常见问题

### Q1: 缺少依赖

```bash
pip install anthropic openai torch torchvision
```

### Q2: GPU不可用

检查CUDA：
```bash
nvidia-smi
```

如果显示正常，但PyTorch检测不到，可能需要安装对应CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q3: API调用失败

检查API Key：
```bash
echo $MIMO_API_KEY
```

测试API连接：
```bash
python3 -c "
import anthropic
client = anthropic.Anthropic()
print('API连接成功')
"
```

### Q4: 内存不足

减小batch_size，编辑 `run_train_server.py`：
```python
args = argparse.Namespace(
    batch_size=16,  # 或更小
    # ...
)
```

### Q5: 训练太慢

减少训练轮数：
```python
args = argparse.Namespace(
    epochs_stage1=5,
    epochs_stage2=20,
    # ...
)
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

# 3. 安装依赖
bash install_server.sh

# 4. 设置API Key
export MIMO_API_KEY="your_key"

# 5. 数据标注
python run_auto_label.py

# 6. 数据集划分
python run_split_dataset.py

# 7. 后台训练
nohup python run_train_server.py > train.log 2>&1 &

# 8. 监控训练
tail -f train.log

# 9. 训练完成后评估
python run_evaluate.py

# 10. 下载结果到本地
```
