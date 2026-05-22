#!/bin/bash
# A40服务器一键运行脚本
# 使用方法: nohup bash run_all_a40.sh > train_all.log 2>&1 &

set -e

# 检查API Key
if [ -z "${MIMO_API_KEY:-}" ]; then
  echo "错误: 请先设置 MIMO_API_KEY"
  echo "示例: export MIMO_API_KEY='your_api_key'"
  exit 1
fi

# 激活conda环境
conda activate a40_env

echo "=========================================="
echo "开始一键运行"
echo "时间: $(date)"
echo "=========================================="

# 步骤1: 数据标注
echo "[1/4] 数据标注..."
python run_auto_label.py

# 步骤2: 数据集划分
echo "[2/4] 数据集划分..."
python run_split_dataset.py

# 步骤3: 训练模型
echo "[3/4] 训练模型..."
python run_train_server.py

# 步骤4: 评估模型
echo "[4/4] 评估模型..."
python run_evaluate.py

echo "=========================================="
echo "全部完成！"
echo "时间: $(date)"
echo "=========================================="
