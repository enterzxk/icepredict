"""
覆冰图像分类器训练脚本
支持两阶段训练：冻结backbone微调 + 全参数微调
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np

from src.ice_dataset import get_dataloaders, LABEL_NAMES, get_label_names
from src.ice_classifier import create_model
from src.data_augmentation import Mixup, Cutmix

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        """tensorboard未安装时的空实现，避免训练入口直接导入失败。"""

        def __init__(self, *args, **kwargs):
            print("警告: 未安装 tensorboard，将跳过TensorBoard日志记录")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    解决类别不平衡问题
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (B, C)
            targets: labels (B, C)
        """
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class MultiLabelMetrics:
    """多标签分类指标计算"""

    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        label_names=None,
    ) -> dict:
        """
        计算每个类别的Precision、Recall、F1

        Args:
            predictions: 预测概率 (B, C)
            targets: 真实标签 (B, C)
            threshold: 分类阈值

        Returns:
            dict: 每个类别的指标
        """
        label_names = list(label_names or LABEL_NAMES)

        # 二值化预测
        pred_binary = (predictions > threshold).float()

        metrics = {}
        for i, label_name in enumerate(label_names):
            tp = ((pred_binary[:, i] == 1) & (targets[:, i] == 1)).sum().float()
            fp = ((pred_binary[:, i] == 1) & (targets[:, i] == 0)).sum().float()
            fn = ((pred_binary[:, i] == 0) & (targets[:, i] == 1)).sum().float()
            tn = ((pred_binary[:, i] == 0) & (targets[:, i] == 0)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

            metrics[label_name] = {
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
                "accuracy": accuracy.item(),
                "tp": tp.item(),
                "fp": fp.item(),
                "fn": fn.item(),
                "tn": tn.item(),
            }

        # 计算总体指标
        total_correct = (pred_binary == targets).all(dim=1).sum().float()
        metrics["overall"] = {
            "exact_match_accuracy": (total_correct / len(targets)).item(),
        }

        return metrics


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    label_names=None,
) -> dict:
    """训练一个epoch"""
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    mixup = Mixup(alpha=1.0) if use_mixup else None
    cutmix = Cutmix(alpha=1.0) if use_cutmix else None

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # 应用Mixup或Cutmix
        if mixup and np.random.random() < 0.5:
            images, targets = mixup(images, targets)
        elif cutmix and np.random.random() < 0.5:
            images, targets = cutmix(images, targets)

        # 前向传播
        logits = model(images)
        loss = criterion(logits, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录
        total_loss += loss.item()
        probabilities = torch.sigmoid(logits)
        all_predictions.append(probabilities.detach().cpu())
        all_targets.append(targets.detach().cpu())

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 计算指标
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    label_names = list(label_names or LABEL_NAMES)
    metrics = MultiLabelMetrics.compute_metrics(all_predictions, all_targets, label_names=label_names)

    avg_loss = total_loss / len(train_loader)

    # 记录到TensorBoard
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    for label_name in label_names:
        writer.add_scalar(f"Train/{label_name}/F1", metrics[label_name]["f1"], epoch)
        writer.add_scalar(f"Train/{label_name}/Precision", metrics[label_name]["precision"], epoch)
        writer.add_scalar(f"Train/{label_name}/Recall", metrics[label_name]["recall"], epoch)

    return {"loss": avg_loss, "metrics": metrics}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    label_names=None,
) -> dict:
    """验证"""
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        probabilities = torch.sigmoid(logits)
        all_predictions.append(probabilities.cpu())
        all_targets.append(targets.cpu())

    # 计算指标
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    label_names = list(label_names or LABEL_NAMES)
    metrics = MultiLabelMetrics.compute_metrics(all_predictions, all_targets, label_names=label_names)

    avg_loss = total_loss / len(val_loader)

    # 记录到TensorBoard
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    for label_name in label_names:
        writer.add_scalar(f"Val/{label_name}/F1", metrics[label_name]["f1"], epoch)
        writer.add_scalar(f"Val/{label_name}/Precision", metrics[label_name]["precision"], epoch)
        writer.add_scalar(f"Val/{label_name}/Recall", metrics[label_name]["recall"], epoch)

    return {"loss": avg_loss, "metrics": metrics}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_path: str,
    metadata: dict = None,
):
    """保存checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler=None,
) -> dict:
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="覆冰图像分类器训练")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="数据目录")
    parser.add_argument("--label-file", type=str, default=None, help="标注文件路径")
    parser.add_argument("--task", type=str, choices=["multi_label", "binary"], default="multi_label", help="任务类型")
    parser.add_argument("--model-name", type=str, default="resnet50", help="模型名称")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--image-size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--epochs-stage1", type=int, default=10, help="第一阶段训练轮数")
    parser.add_argument("--epochs-stage2", type=int, default=30, help="第二阶段训练轮数")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="第一阶段学习率")
    parser.add_argument("--lr-stage2", type=float, default=1e-4, help="第二阶段学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout比率")
    parser.add_argument("--use-focal-loss", action="store_true", help="使用Focal Loss")
    parser.add_argument("--use-mixup", action="store_true", help="使用Mixup增强")
    parser.add_argument("--use-cutmix", action="store_true", help="使用Cutmix增强")
    parser.add_argument("--no-weighted-sampling", action="store_true", help="关闭WeightedRandomSampler")
    parser.add_argument("--disable-pos-weight", action="store_true", help="关闭BCEWithLogitsLoss的pos_weight")
    parser.add_argument("--num-workers", type=int, default=4, help="工作线程数")
    parser.add_argument("--save-dir", type=str, default="weights/ice_classifier", help="保存目录")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="仅加载模型权重作为初始化，不恢复优化器和epoch")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--device", type=str, default=None, help="设备")
    parser.add_argument("--tensorboard-dir", type=str, default="runs", help="TensorBoard日志目录")
    args = parser.parse_args(argv)

    # 设备
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    label_names = get_label_names(args.task)
    num_classes = len(label_names)
    checkpoint_metadata = {
        "task": args.task,
        "label_names": label_names,
        "num_classes": num_classes,
    }
    print(f"任务类型: {args.task}")
    print(f"标签: {label_names}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, f"ice_classifier_{timestamp}"))

    # 数据加载
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        label_file=args.label_file,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_strong_augment=True,
        use_weighted_sampling=not args.no_weighted_sampling,
        label_names=label_names,
    )

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=not bool(args.init_checkpoint),
        dropout=args.dropout,
        freeze_backbone=True,  # 第一阶段冻结backbone
    )
    model = model.to(device)
    if args.init_checkpoint:
        print(f"\n加载初始化权重: {args.init_checkpoint}")
        load_checkpoint(args.init_checkpoint, model)

    # 损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("使用Focal Loss")
    else:
        pos_weight = None
        if not args.disable_pos_weight and hasattr(train_loader.dataset, "get_label_weights"):
            pos_weight = train_loader.dataset.get_label_weights().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None:
            print(f"使用BCE Loss，pos_weight={pos_weight.detach().cpu().tolist()}")
        else:
            print("使用BCE Loss")

    # 优化器
    optimizer = optim.AdamW(
        model.get_param_groups(lr=args.lr_stage1, weight_decay=args.weight_decay),
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # 恢复训练
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume:
        print(f"\n恢复训练: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint.get("metrics", {}).get("val_best_f1", 0.0)

    # ===== 第一阶段：冻结backbone训练 =====
    print("\n" + "="*60)
    print("第一阶段：冻结backbone，只训练分类头")
    print("="*60)

    for epoch in range(start_epoch, args.epochs_stage1):
        print(f"\nEpoch [{epoch+1}/{args.epochs_stage1}]")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, args.use_mixup, args.use_cutmix, label_names,
        )
        print(f"  训练 Loss: {train_metrics['loss']:.4f}")

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, label_names)
        print(f"  验证 Loss: {val_metrics['loss']:.4f}")

        # 打印每个类别的F1
        for label_name in label_names:
            train_f1 = train_metrics["metrics"][label_name]["f1"]
            val_f1 = val_metrics["metrics"][label_name]["f1"]
            print(f"    {label_name}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

        # 计算平均F1
        avg_val_f1 = np.mean([val_metrics["metrics"][ln]["f1"] for ln in label_names])

        # 保存最佳模型
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"val_best_f1": best_val_f1},
                os.path.join(args.save_dir, "best_stage1.pth"),
                checkpoint_metadata,
            )
            print(f"  保存最佳模型 (F1={best_val_f1:.4f})")

        # 保存最新模型
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {"val_best_f1": best_val_f1},
            os.path.join(args.save_dir, "latest_stage1.pth"),
            checkpoint_metadata,
        )

        # 更新学习率
        scheduler.step()

    # ===== 第二阶段：解冻backbone微调 =====
    print("\n" + "="*60)
    print("第二阶段：解冻backbone，全参数微调")
    print("="*60)

    # 解冻backbone
    model.unfreeze_backbone()

    # 重新创建优化器（使用较小的学习率）
    optimizer = optim.AdamW(
        model.get_param_groups(lr=args.lr_stage2, weight_decay=args.weight_decay),
        weight_decay=args.weight_decay,
    )

    # 重新创建学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)

    # 重置best_f1
    best_val_f1 = 0.0

    for epoch in range(args.epochs_stage2):
        print(f"\nEpoch [{epoch+1}/{args.epochs_stage2}]")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, args.use_mixup, args.use_cutmix, label_names,
        )
        print(f"  训练 Loss: {train_metrics['loss']:.4f}")

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, label_names)
        print(f"  验证 Loss: {val_metrics['loss']:.4f}")

        # 打印每个类别的F1
        for label_name in label_names:
            train_f1 = train_metrics["metrics"][label_name]["f1"]
            val_f1 = val_metrics["metrics"][label_name]["f1"]
            print(f"    {label_name}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

        # 计算平均F1
        avg_val_f1 = np.mean([val_metrics["metrics"][ln]["f1"] for ln in label_names])

        # 保存最佳模型
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"val_best_f1": best_val_f1},
                os.path.join(args.save_dir, "best_stage2.pth"),
                checkpoint_metadata,
            )
            print(f"  保存最佳模型 (F1={best_val_f1:.4f})")

        # 保存最新模型
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {"val_best_f1": best_val_f1},
            os.path.join(args.save_dir, "latest_stage2.pth"),
            checkpoint_metadata,
        )

        # 更新学习率
        scheduler.step()

    # ===== 测试 =====
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)

    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, "best_stage2.pth")
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)
        print(f"加载最佳模型: {best_model_path}")

    test_metrics = validate(model, test_loader, criterion, device, 0, writer, label_names)
    print(f"\n测试集结果:")
    for label_name in label_names:
        metrics = test_metrics["metrics"][label_name]
        print(f"  {label_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")

    # 保存测试结果
    results = {
        "test_metrics": test_metrics["metrics"],
        "best_val_f1": best_val_f1,
        "label_names": label_names,
        "config": vars(args),
    }
    results_file = os.path.join(args.save_dir, "training_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n训练结果已保存至: {results_file}")

    # 关闭TensorBoard
    writer.close()
    print("\n训练完成!")


if __name__ == "__main__":
    main()
