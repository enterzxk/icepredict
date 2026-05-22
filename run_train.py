"""
训练脚本 - 直接运行版本
无需命令行参数，直接在PyCharm中运行
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse


def _build_train_argv(args):
    argv = [
        "--data-dir", args.data_dir,
        "--task", args.task,
        "--model-name", args.model_name,
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--epochs-stage1", str(args.epochs_stage1),
        "--epochs-stage2", str(args.epochs_stage2),
        "--lr-stage1", str(args.lr_stage1),
        "--lr-stage2", str(args.lr_stage2),
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--num-workers", str(args.num_workers),
        "--save-dir", args.save_dir,
        "--tensorboard-dir", args.tensorboard_dir,
    ]
    if getattr(args, "init_checkpoint", None):
        argv.extend(["--init-checkpoint", args.init_checkpoint])
    if args.label_file:
        argv.extend(["--label-file", args.label_file])
    if args.resume:
        argv.extend(["--resume", args.resume])
    if args.device:
        argv.extend(["--device", args.device])
    if args.use_focal_loss:
        argv.append("--use-focal-loss")
    if args.use_mixup:
        argv.append("--use-mixup")
    if args.use_cutmix:
        argv.append("--use-cutmix")
    if getattr(args, "no_weighted_sampling", False):
        argv.append("--no-weighted-sampling")
    if getattr(args, "disable_pos_weight", False):
        argv.append("--disable-pos-weight")
    return argv

def main():
    # 配置参数（可在PyCharm中直接修改）
    args = argparse.Namespace(
        data_dir=".",
        label_file="data/binary_dataset",
        task="binary",
        model_name="resnet50",
        batch_size=32,
        image_size=224,
        epochs_stage1=10,
        epochs_stage2=30,
        lr_stage1=1e-3,
        lr_stage2=1e-4,
        weight_decay=1e-4,
        dropout=0.5,
        use_focal_loss=False,
        use_mixup=False,
        use_cutmix=False,
        no_weighted_sampling=False,
        disable_pos_weight=False,
        num_workers=4,
        save_dir="weights/ice_binary_classifier",
        init_checkpoint=None,
        resume=None,
        device=None,  # 自动检测GPU/CPU
        tensorboard_dir="runs_binary",
    )

    # 运行训练
    print("="*60)
    print("覆冰图像分类器训练")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"标注目录: {args.label_file}")
    print(f"任务类型: {args.task}")
    print(f"模型: {args.model_name}")
    print(f"批次大小: {args.batch_size}")
    print(f"第一阶段轮数: {args.epochs_stage1}")
    print(f"第二阶段轮数: {args.epochs_stage2}")
    print(f"Focal Loss: {args.use_focal_loss}")
    print(f"Mixup: {args.use_mixup}")
    print("="*60)

    if args.label_file and not os.path.exists(args.label_file):
        print(f"错误: 标注划分目录不存在: {args.label_file}")
        print("请先运行 run_split_dataset.py")
        return

    from train_ice_classifier import main as train_main

    train_main(_build_train_argv(args))

if __name__ == "__main__":
    main()
