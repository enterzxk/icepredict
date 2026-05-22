#!/bin/bash
# 一键运行完整流程
# 使用方法: nohup bash run_pipeline.sh > pipeline.log 2>&1 &

# 检查API Key
if [ -z "${MIMO_API_KEY:-}" ]; then
  echo "错误: 请先设置 MIMO_API_KEY"
  echo "示例: export MIMO_API_KEY='your_api_key'"
  exit 1
fi

# 激活环境
conda activate a40_env

echo "===== 开始运行 ====="
echo "时间: $(date)"

# 1. 数据标注
echo "[1/5] 数据标注..."
python run_auto_label.py

# 2. 数据集划分
echo "[2/5] 数据集划分..."
python run_split_dataset.py

# 3. 训练模型
echo "[3/5] 训练模型..."
python run_train_server.py

# 4. 评估模型
echo "[4/5] 评估模型..."
python run_evaluate.py

# 5. 推理测试
echo "[5/5] 推理测试..."
python run_inference.py

echo "===== 全部完成 ====="
echo "时间: $(date)"
