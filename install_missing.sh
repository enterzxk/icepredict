#!/bin/bash
# 安装缺失依赖脚本
# 适用于已有PyTorch环境的服务器

set -e

echo "=========================================="
echo "安装缺失依赖"
echo "=========================================="

# 激活conda环境
echo "激活 a40_env 环境..."
conda activate a40_env

# 安装缺失的包
echo "安装缺失的Python包..."
pip install einops kornia openai anthropic Pillow tensorboard

# 验证安装
echo ""
echo "验证安装..."
python3 -c "
import einops
print(f'einops: {einops.__version__}')
"

python3 -c "
import kornia
print(f'kornia: {kornia.__version__}')
"

python3 -c "
import openai
print(f'openai: {openai.__version__}')
"

python3 -c "
import anthropic
print(f'anthropic: {anthropic.__version__}')
"

python3 -c "
from PIL import Image
print(f'Pillow: OK')
"

python3 -c "
import tensorboard
print(f'tensorboard: {tensorboard.__version__}')
"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "接下来需要设置API Key："
echo "  export MIMO_API_KEY='your_api_key'"
echo ""
echo "然后可以运行以下脚本："
echo "  1. python run_auto_label.py    # 数据标注"
echo "  2. python run_split_dataset.py # 数据集划分"
echo "  3. python run_train_server.py  # 训练模型"
echo "  4. python run_evaluate.py      # 评估模型"
echo "  5. python run_inference.py     # 推理测试"
