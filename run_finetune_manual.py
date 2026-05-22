"""
使用 644 张人工标注真实图片继续微调二分类模型。

从 Roboflow 训练得到的 best_stage2.pth 初始化，只用小学习率在真实数据上适配。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _build_train_argv(args):
    argv = [
        "--data-dir", args.data_dir,
        "--label-file", args.label_file,
        "--task", "binary",
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
        "--init-checkpoint", args.init_checkpoint,
    ]
    if args.device:
        argv.extend(["--device", args.device])
    if args.use_focal_loss:
        argv.append("--use-focal-loss")
    if args.use_mixup:
        argv.append("--use-mixup")
    if args.use_cutmix:
        argv.append("--use-cutmix")
    if args.no_weighted_sampling:
        argv.append("--no-weighted-sampling")
    if args.disable_pos_weight:
        argv.append("--disable-pos-weight")
    return argv


def main():
    args = argparse.Namespace(
        data_dir=".",
        label_file="data/manual_finetune",
        init_checkpoint="weights/ice_binary_classifier/best_stage2.pth",
        model_name="resnet50",
        batch_size=16,
        image_size=224,
        epochs_stage1=8,
        epochs_stage2=3,
        lr_stage1=3e-5,
        lr_stage2=3e-6,
        weight_decay=1e-4,
        dropout=0.5,
        use_focal_loss=False,
        use_mixup=False,
        use_cutmix=False,
        no_weighted_sampling=True,
        disable_pos_weight=True,
        num_workers=4,
        save_dir="weights/ice_binary_manual_finetune_v2",
        device="cuda:0",
        tensorboard_dir="runs_manual_finetune_v2",
    )

    if not os.path.exists(args.label_file):
        print(f"错误: 微调数据目录不存在: {args.label_file}")
        print("请先运行 run_prepare_manual_finetune.py")
        return
    if not os.path.exists(args.init_checkpoint):
        print(f"错误: 初始化checkpoint不存在: {args.init_checkpoint}")
        return

    print("=" * 60)
    print("真实监控图二分类微调")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"标注目录: {args.label_file}")
    print(f"初始化权重: {args.init_checkpoint}")
    print(f"保存目录: {args.save_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"epochs: {args.epochs_stage1} + {args.epochs_stage2}")
    print(f"lr: {args.lr_stage1} / {args.lr_stage2}")
    print(f"关闭WeightedRandomSampler: {args.no_weighted_sampling}")
    print(f"关闭pos_weight: {args.disable_pos_weight}")
    print("=" * 60)

    from train_ice_classifier import main as train_main

    train_main(_build_train_argv(args))


if __name__ == "__main__":
    main()
