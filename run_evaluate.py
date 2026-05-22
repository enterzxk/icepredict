"""
评估脚本 - 直接运行版本
无需命令行参数，直接在PyCharm中运行
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse


def _build_eval_argv(args):
    argv = [
        "--checkpoint", args.checkpoint,
        "--data-dir", args.data_dir,
        "--task", args.task,
        "--model-name", args.model_name,
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--threshold", str(args.threshold),
        "--num-workers", str(args.num_workers),
        "--save-dir", args.save_dir,
    ]
    if args.label_file:
        argv.extend(["--label-file", args.label_file])
    if args.device:
        argv.extend(["--device", args.device])
    return argv

def main():
    # 配置参数（可在PyCharm中直接修改）
    args = argparse.Namespace(
        checkpoint="weights/ice_binary_classifier/best_stage2.pth",
        data_dir=".",
        label_file="data/binary_dataset",
        task="binary",
        model_name="resnet50",
        batch_size=32,
        image_size=224,
        threshold=0.5,
        num_workers=4,
        save_dir="experiments/eval_binary",
        device=None,  # 自动检测GPU/CPU
    )

    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型checkpoint不存在: {args.checkpoint}")
        print("请先运行 run_train.py 训练模型")
        return

    # 运行评估
    print("="*60)
    print("覆冰图像分类器评估")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"数据目录: {args.data_dir}")
    print(f"标注目录: {args.label_file}")
    print(f"任务类型: {args.task}")
    print(f"阈值: {args.threshold}")
    print("="*60)

    from evaluate_ice_classifier import main as eval_main

    eval_main(_build_eval_argv(args))

if __name__ == "__main__":
    main()
