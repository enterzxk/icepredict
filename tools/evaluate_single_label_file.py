"""
评估单个二分类 label JSON 文件。

适合自有 644 张人工验证集，不要求 train/val/test 三份划分文件都存在。
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_binary_metrics(targets: List[int], probabilities: List[float], threshold: float) -> Dict:
    tp = fp = fn = tn = 0
    for target, probability in zip(targets, probabilities):
        pred = int(probability >= threshold)
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive": sum(targets),
        "negative": len(targets) - sum(targets),
    }


def load_label_items(label_file: str) -> List[Dict]:
    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"label_file 必须是列表JSON: {label_file}")
    return data


def evaluate_single_label_file(args) -> Dict:
    import torch
    from PIL import Image

    from src.data_augmentation import get_val_transforms
    from src.ice_classifier import load_model
    from tools.export_error_samples import _build_filename_index, resolve_image_path

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_label_items(args.label_file)
    data_dir = Path(args.data_dir)
    filename_index = _build_filename_index([
        data_dir,
        data_dir / "data" / "imagine",
        data_dir / "data" / "roboflow_train",
        Path("data/imagine"),
        Path("data/roboflow_train"),
    ])

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_classes=1,
        device=device,
    )
    model.eval()
    transform = get_val_transforms(image_size=args.image_size)

    rows = []
    images = []
    meta = []
    missing = []

    def flush_batch():
        if not images:
            return
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).detach().cpu().numpy().reshape(-1)
        for item_meta, prob in zip(meta, probs):
            target = item_meta["target"]
            pred = int(float(prob) >= args.threshold)
            if target == 1 and pred == 1:
                error_type = "TP"
            elif target == 0 and pred == 1:
                error_type = "FP"
            elif target == 1 and pred == 0:
                error_type = "FN"
            else:
                error_type = "TN"
            rows.append({
                "image_path": item_meta["image_path"],
                "resolved_path": item_meta["resolved_path"],
                "true_label": target,
                "pred_prob": f"{float(prob):.8f}",
                "threshold": args.threshold,
                "pred_label": pred,
                "error_type": error_type,
            })
        images.clear()
        meta.clear()

    for item in items:
        image_path = item.get("image_path", "")
        resolved_path = resolve_image_path(image_path, data_dir, filename_index)
        if not resolved_path:
            missing.append(image_path)
            continue
        vector = item.get("label_vector", [0])
        target = int(bool(vector[0]))
        image = Image.open(resolved_path).convert("RGB")
        images.append(transform(image))
        meta.append({
            "image_path": image_path,
            "resolved_path": resolved_path,
            "target": target,
        })
        if len(images) >= args.batch_size:
            flush_batch()
    flush_batch()

    targets = [int(row["true_label"]) for row in rows]
    probabilities = [float(row["pred_prob"]) for row in rows]
    metrics = compute_binary_metrics(targets, probabilities, args.threshold)

    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "image_path", "resolved_path", "true_label", "pred_prob", "threshold", "pred_label", "error_type"
        ])
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "metrics": {"冰雪异常": metrics},
        "label_file": args.label_file,
        "threshold": args.threshold,
        "count": len(rows),
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "config": vars(args),
    }
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    report = f"""
冰雪二分类单文件评估报告
============================================================

样本数: {len(rows)}
正样本: {metrics['positive']}
负样本: {metrics['negative']}
阈值: {args.threshold}
找不到图片: {len(missing)}

冰雪异常:
  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.1f}%)
  Recall:    {metrics['recall']:.4f} ({metrics['recall'] * 100:.1f}%)
  F1-Score:  {metrics['f1']:.4f} ({metrics['f1'] * 100:.1f}%)
  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)
  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}

--- Markdown表格 ---

| 类别 | Precision | Recall | F1-Score | Accuracy |
|------|-----------|--------|----------|----------|
| 冰雪异常 | {metrics['precision'] * 100:.1f}% | {metrics['recall'] * 100:.1f}% | {metrics['f1'] * 100:.1f}% | {metrics['accuracy'] * 100:.1f}% |
"""
    with open(output_dir / "eval_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"评估结果已保存至: {output_dir}")
    print(f"逐图预测已保存至: {csv_path}")
    if missing:
        print(f"警告: 找不到 {len(missing)} 张图片，示例: {missing[:5]}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估单个二分类标签JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--task", choices=["binary"], default="binary")
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default="experiments/eval_single")
    args = parser.parse_args(argv)
    evaluate_single_label_file(args)


if __name__ == "__main__":
    main()
