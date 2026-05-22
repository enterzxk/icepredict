#!/bin/bash
# 一键运行脚本 - 后台执行所有步骤
# 使用方法: nohup bash run_all.sh > all.log 2>&1 &

set -e

echo "=========================================="
echo "IceGuard AI - 一键运行"
echo "=========================================="
echo "开始时间: $(date)"

# 检查环境变量
if [ -z "${MIMO_API_KEY:-}" ]; then
  echo "错误: 请先设置 MIMO_API_KEY"
  echo "示例: export MIMO_API_KEY='your_api_key'"
  exit 1
fi

# 激活conda环境
echo "激活 a40_env 环境..."
source ~/miniconda3/etc/profile.d/conda.sh  # 根据实际路径调整
conda activate a40_env

# 步骤1: 数据标注
echo ""
echo "=========================================="
echo "步骤1: 数据标注"
echo "=========================================="
python run_auto_label.py

# 步骤2: 数据集划分
echo ""
echo "=========================================="
echo "步骤2: 数据集划分"
echo "=========================================="
python run_split_dataset.py

# 步骤3: 训练模型
echo ""
echo "=========================================="
echo "步骤3: 训练模型"
echo "=========================================="
python run_train_server.py

# 步骤4: 评估模型
echo ""
echo "=========================================="
echo "步骤4: 评估模型"
echo "=========================================="
python run_evaluate.py

echo ""
echo "=========================================="
echo "所有步骤完成！"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "结果文件："
echo "  - 标注结果: multi_label_results.json"
echo "  - 训练标签: data/labels/training_labels.json"
echo "  - 数据集: data/dataset/"
echo "  - 模型权重: weights/ice_classifier/"
echo "  - 评估结果: experiments/eval_figures/"
