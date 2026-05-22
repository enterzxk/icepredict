"""
服务器训练脚本 - 优化版
适用于GPU服务器环境，使用更大batch_size和更多训练轮数
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
    # 服务器优化配置
    args = argparse.Namespace(
        data_dir=".",
        label_file="data/binary_dataset",
        task="binary",
        model_name="resnet50",
        batch_size=64,              # 服务器GPU内存大，使用更大batch
        image_size=224,
        epochs_stage1=10,
        epochs_stage2=50,           # 服务器可以训练更多轮
        lr_stage1=1e-3,
        lr_stage2=1e-4,
        weight_decay=1e-4,
        dropout=0.5,
        use_focal_loss=False,
        use_mixup=False,
        use_cutmix=False,
        no_weighted_sampling=False,
        disable_pos_weight=False,
        num_workers=8,              # 服务器CPU核心多
        save_dir="weights/ice_binary_classifier",
        init_checkpoint=None,
        resume=None,
        device="cuda:0",            # 明确指定GPU
        tensorboard_dir="runs_binary",
    )

    # 打印配置
    print("="*60)
    print("覆冰图像分类器训练 - 服务器版")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"标注目录: {args.label_file}")
    print(f"任务类型: {args.task}")
    print(f"模型: {args.model_name}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像尺寸: {args.image_size}")
    print(f"第一阶段轮数: {args.epochs_stage1}")
    print(f"第二阶段轮数: {args.epochs_stage2}")
    print(f"学习率: {args.lr_stage1} / {args.lr_stage2}")
    print(f"Focal Loss: {args.use_focal_loss}")
    print(f"Mixup: {args.use_mixup}")
    print(f"Cutmix: {args.use_cutmix}")
    print(f"工作线程: {args.num_workers}")
    print(f"设备: {args.device}")
    print("="*60)

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        print("请先运行数据准备脚本：")
        print("  1. python run_auto_label.py")
        print("  2. python run_split_dataset.py")
        return
    if args.label_file and not os.path.exists(args.label_file):
        print(f"错误: 标注划分目录不存在: {args.label_file}")
        print("请先运行 run_split_dataset.py 生成 train_labels.json / val_labels.json / test_labels.json")
        return

    # 检查数据量
    train_dir = os.path.join(args.data_dir, "train")
    if os.path.exists(train_dir):
        total_images = sum(
            len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
            for _, _, files in os.walk(train_dir)
        )
        print(f"训练集图像数量: {total_images}")
        if total_images < 100:
            print("警告: 训练数据较少，建议先运行数据标注扩充数据集")

    # 运行训练
    from train_ice_classifier import main as train_main

    train_main(_build_train_argv(args))

if __name__ == "__main__":
    main()
