#!/bin/bash
# 服务器环境安装脚本
# 使用方法: bash setup_server.sh

set -e

echo "=========================================="
echo "IceGuard AI - 服务器环境安装"
echo "=========================================="

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 检查GPU
echo "检查GPU环境..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'CUDA版本: {torch.version.cuda}')
else:
    print('警告: 未检测到GPU，将使用CPU训练（速度较慢）')
"

# 创建必要目录
echo "创建目录结构..."
mkdir -p data/labels
mkdir -p data/dataset
mkdir -p weights/ice_classifier
mkdir -p runs
mkdir -p experiments/eval_figures
mkdir -p tools

echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "接下来可以运行以下脚本："
echo "  1. python run_auto_label.py    # 数据标注"
echo "  2. python run_split_dataset.py # 数据集划分"
echo "  3. python run_train.py         # 训练模型"
echo "  4. python run_evaluate.py      # 评估模型"
echo "  5. python run_inference.py     # 推理测试"
