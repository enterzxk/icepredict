"""
覆冰图像分类器评估脚本
计算每个类别的P/R/F1，生成混淆矩阵和ROC曲线
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)

from src.ice_dataset import get_dataloaders, LABEL_NAMES, get_label_names
from src.ice_classifier import load_model
from src.data_augmentation import get_val_transforms

# 中文配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str,
    threshold: float = 0.5,
    label_names=None,
) -> dict:
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        threshold: 分类阈值

    Returns:
        dict: 评估结果
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_predictions.append(probabilities.cpu())
            all_targets.append(targets.cpu())
            all_probabilities.append(probabilities.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_probabilities = torch.cat(all_probabilities)

    label_names = list(label_names or LABEL_NAMES)

    # 二值化预测
    pred_binary = (all_predictions > threshold).float()

    # 计算每个类别的指标
    metrics = {}
    for i, label_name in enumerate(label_names):
        tp = ((pred_binary[:, i] == 1) & (all_targets[:, i] == 1)).sum().float()
        fp = ((pred_binary[:, i] == 1) & (all_targets[:, i] == 0)).sum().float()
        fn = ((pred_binary[:, i] == 0) & (all_targets[:, i] == 1)).sum().float()
        tn = ((pred_binary[:, i] == 0) & (all_targets[:, i] == 0)).sum().float()

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
    total_correct = (pred_binary == all_targets).all(dim=1).sum().float()
    metrics["overall"] = {
        "exact_match_accuracy": (total_correct / len(all_targets)).item(),
        "hamming_loss": (pred_binary != all_targets).float().mean().item(),
    }

    return {
        "metrics": metrics,
        "predictions": all_predictions.numpy(),
        "targets": all_targets.numpy(),
        "probabilities": all_probabilities.numpy(),
    }


def plot_confusion_matrices(
    targets: np.ndarray,
    predictions: np.ndarray,
    save_dir: str,
    threshold: float = 0.5,
    label_names=None,
):
    """绘制每个类别的混淆矩阵"""
    label_names = list(label_names or LABEL_NAMES)
    pred_binary = (predictions > threshold).astype(int)

    cols = min(2, len(label_names))
    rows = int(np.ceil(len(label_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, label_name in enumerate(label_names):
        cm = confusion_matrix(targets[:, i], pred_binary[:, i], labels=[0, 1])

        ax = axes[i]
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['无', '有'], fontsize=11)
        ax.set_yticklabels(['无', '有'], fontsize=11)
        ax.set_xlabel('预测', fontsize=12)
        ax.set_ylabel('真实', fontsize=12)
        ax.set_title(f'{label_name} 混淆矩阵', fontsize=13, fontweight='bold')

        # 添加数值
        for j in range(2):
            for k in range(2):
                color = 'white' if cm[j, k] > cm.max() / 2 else 'black'
                ax.text(k, j, str(cm[j, k]), ha='center', va='center',
                       fontsize=16, fontweight='bold', color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes[len(label_names):]:
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")


def plot_roc_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    save_dir: str,
    label_names=None,
):
    """绘制每个类别的ROC曲线"""
    label_names = list(label_names or LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, label_name in enumerate(label_names):
        if len(np.unique(targets[:, i])) < 2:
            print(f"跳过 {label_name} ROC曲线：测试集中只有单一真实类别")
            continue
        fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{label_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (FPR)', fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontsize=12)
    ax.set_title('ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")


def plot_pr_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    save_dir: str,
    label_names=None,
):
    """绘制每个类别的PR曲线"""
    label_names = list(label_names or LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, label_name in enumerate(label_names):
        if len(np.unique(targets[:, i])) < 2:
            print(f"跳过 {label_name} PR曲线：测试集中只有单一真实类别")
            continue
        precision, recall, _ = precision_recall_curve(targets[:, i], probabilities[:, i])
        pr_auc = auc(recall, precision)

        ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{label_name} (AP = {pr_auc:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率 (Recall)', fontsize=12)
    ax.set_ylabel('精确率 (Precision)', fontsize=12)
    ax.set_title('PR曲线', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'pr_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"PR曲线已保存至: {save_path}")


def plot_metrics_bar(metrics: dict, save_dir: str, label_names=None):
    """绘制指标条形图"""
    label_names = list(label_names or LABEL_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 准备数据
    labels = label_names
    precisions = [metrics[l]["precision"] for l in labels]
    recalls = [metrics[l]["recall"] for l in labels]
    f1s = [metrics[l]["f1"] for l in labels]

    base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    # Precision条形图
    bars = axes[0].bar(labels, precisions, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, precisions):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('精确率', fontsize=12)
    axes[0].set_title('Precision', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)

    # Recall条形图
    bars = axes[1].bar(labels, recalls, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, recalls):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('召回率', fontsize=12)
    axes[1].set_title('Recall', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)

    # F1条形图
    bars = axes[2].bar(labels, f1s, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, f1s):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('F1分数', fontsize=12)
    axes[2].set_title('F1-Score', fontsize=13, fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_bar.png')
    plt.savefig(save_path)
    plt.close()
    print(f"指标条形图已保存至: {save_path}")


def generate_report(metrics: dict, save_dir: str, label_names=None, task: str = "multi_label"):
    """生成评估报告"""
    label_names = list(label_names or LABEL_NAMES)
    task_name = "二分类" if task == "binary" else "多标签分类"
    class_text = "、".join(label_names)
    report = f"""
覆冰图像分类器 — 评估报告
{'='*60}

模型: ResNet50 {task_name}
类别: {class_text}

--- 各类别指标 ---

"""
    for label_name in label_names:
        m = metrics[label_name]
        report += f"""
{label_name}:
  Precision: {m['precision']:.4f} ({m['precision']*100:.1f}%)
  Recall:    {m['recall']:.4f} ({m['recall']*100:.1f}%)
  F1-Score:  {m['f1']:.4f} ({m['f1']*100:.1f}%)
  Accuracy:  {m['accuracy']:.4f} ({m['accuracy']*100:.1f}%)
  TP={int(m['tp'])}, FP={int(m['fp'])}, FN={int(m['fn'])}, TN={int(m['tn'])}
"""

    report += f"""
--- 总体指标 ---

Exact Match Accuracy: {metrics['overall']['exact_match_accuracy']:.4f} ({metrics['overall']['exact_match_accuracy']*100:.1f}%)
Hamming Loss: {metrics['overall']['hamming_loss']:.4f}

--- Markdown表格 ---

| 类别 | Precision | Recall | F1-Score | Accuracy |
|------|-----------|--------|----------|----------|
"""
    for label_name in label_names:
        m = metrics[label_name]
        report += f"| {label_name} | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']*100:.1f}% | {m['accuracy']*100:.1f}% |\n"

    report += f"| **总体** | - | - | - | {metrics['overall']['exact_match_accuracy']*100:.1f}% |\n"

    save_path = os.path.join(save_dir, 'eval_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"评估报告已保存至: {save_path}")
    print(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="覆冰图像分类器评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="数据目录")
    parser.add_argument("--label-file", type=str, default=None, help="标注文件路径")
    parser.add_argument("--task", type=str, choices=["multi_label", "binary"], default="multi_label", help="任务类型")
    parser.add_argument("--model-name", type=str, default="resnet50", help="模型名称")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--image-size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    parser.add_argument("--num-workers", type=int, default=4, help="工作线程数")
    parser.add_argument("--save-dir", type=str, default="experiments/eval_figures", help="保存目录")
    parser.add_argument("--device", type=str, default=None, help="设备")
    args = parser.parse_args(argv)

    # 设备
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    label_names = get_label_names(args.task)
    num_classes = len(label_names)
    print(f"任务类型: {args.task}")
    print(f"标签: {label_names}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    print("\n加载模型...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_classes=num_classes,
        device=device,
    )

    # 加载数据
    print("\n加载数据...")
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        label_file=args.label_file,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_strong_augment=False,
        use_weighted_sampling=False,
        label_names=label_names,
    )

    # 评估
    print("\n评估模型...")
    results = evaluate_model(model, test_loader, device, args.threshold, label_names)

    # 生成图表
    print("\n生成图表...")
    plot_confusion_matrices(results["targets"], results["predictions"], args.save_dir, args.threshold, label_names)
    plot_roc_curves(results["targets"], results["probabilities"], args.save_dir, label_names)
    plot_pr_curves(results["targets"], results["probabilities"], args.save_dir, label_names)
    plot_metrics_bar(results["metrics"], args.save_dir, label_names)

    # 生成报告
    generate_report(results["metrics"], args.save_dir, label_names, args.task)

    # 保存结果
    save_results = {
        "metrics": results["metrics"],
        "label_names": label_names,
        "config": vars(args),
    }
    results_file = os.path.join(args.save_dir, "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存至: {results_file}")


if __name__ == "__main__":
    main()
