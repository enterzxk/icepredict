"""
准备 644 张人工标注数据的微调划分。

先把 manual_labels.csv 转成二分类 JSON，再划分 train/val/test。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.convert_manual_eval import convert_manual_csv
from tools.split_dataset import load_labels, stratified_split, save_split_info


MANUAL_CSV = "manual_labels.csv"
MANUAL_EVAL_DIR = "data/manual_eval"
OUTPUT_DIR = "data/manual_finetune"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def main():
    if not os.path.exists(MANUAL_CSV):
        print(f"错误: 找不到人工标注文件: {MANUAL_CSV}")
        print("请先把桌面的 manual_labels.csv 上传到项目根目录")
        return

    print("=" * 60)
    print("准备自有 644 张人工数据微调集")
    print("=" * 60)
    summary = convert_manual_csv(MANUAL_CSV, MANUAL_EVAL_DIR)
    label_file = os.path.join(MANUAL_EVAL_DIR, "manual_labels_binary.json")
    data = load_labels(label_file)

    train_data, val_data, test_data = stratified_split(
        data,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        SEED,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_split_info(train_data, val_data, test_data, OUTPUT_DIR)

    print("\n人工数据微调集准备完成")
    print(f"CSV编码: {summary['encoding']}")
    print(f"总样本数: {summary['total']}")
    print(f"冰雪异常: {summary['positive']}")
    print(f"正常: {summary['negative']}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
