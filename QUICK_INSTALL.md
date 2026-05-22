# 快速安装指南（已有PyTorch环境）

## 环境情况

- 服务器：A40 GPU
- 虚拟环境：`a40_env`（conda）
- 已有：PyTorch、conda
- 缺失：einops、kornia、openai、anthropic、Pillow、tensorboard

## 安装步骤

### 1. 上传代码到服务器

PyCharm：`Tools → Deployment → Upload to...`

### 2. SSH连接服务器

```bash
ssh user@server
cd /path/to/project
```

### 3. 激活环境并安装缺失包

```bash
conda activate a40_env
bash install_missing.sh
```

或者手动安装：

```bash
conda activate a40_env
pip install einops kornia openai anthropic Pillow tensorboard
```

### 4. 设置API Key

```bash
export MIMO_API_KEY="your_mimo_api_key"

# 永久生效
echo 'export MIMO_API_KEY="your_mimo_api_key"' >> ~/.bashrc
source ~/.bashrc
```

### 5. 运行脚本

```bash
conda activate a40_env

python run_auto_label.py      # 数据标注
python run_split_dataset.py   # 数据集划分
python run_train_server.py    # 训练模型
python run_evaluate.py        # 评估模型
python run_inference.py       # 推理测试
```

## 完整流程

```bash
# 1. 连接服务器
ssh user@server

# 2. 进入项目
cd /path/to/project

# 3. 激活环境
conda activate a40_env

# 4. 安装缺失包
pip install einops kornia openai anthropic Pillow tensorboard

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
nvidia-smi
```

## 常见问题

### Q: conda activate 不生效

```bash
# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
# 或
source ~/anaconda3/etc/profile.d/conda.sh

# 然后激活
conda activate a40_env
```

### Q: pip install 很慢

使用国内镜像源：

```bash
pip install einops kornia openai anthropic Pillow tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: API调用失败

检查API Key：

```bash
echo $MIMO_API_KEY
```

测试连接：

```bash
python -c "import anthropic; client = anthropic.Anthropic(); print('OK')"
```

## 后台训练

```bash
# 启动
nohup python run_train_server.py > train.log 2>&1 &

# 查看日志
tail -f train.log

# 查看GPU
watch -n 1 nvidia-smi

# 停止
kill $(pgrep -f run_train_server)
```
