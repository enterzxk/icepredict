"""
导出二分类模型的漏报/误报样本。

默认分析测试集，将 FN/FP 图片复制到单独文件夹，并生成 CSV，方便人工查看。
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify_binary_error(true_label: int, probability: float, threshold: float) -> str:
    """返回 TP/FP/FN/TN。"""
    pred_label = int(probability >= threshold)
    true_label = int(bool(true_label))
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 1:
        return "FP"
    if true_label == 1 and pred_label == 0:
        return "FN"
    return "TN"


def _filename_from_any_path(image_path: str) -> str:
    return PureWindowsPath(image_path).name or Path(image_path).name


def _build_filename_index(roots: Iterable[Path]) -> Dict[str, str]:
    index = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(path.name, str(path))
    return index


def resolve_image_path(image_path: str, data_dir: Path, filename_index: Dict[str, str]) -> Optional[str]:
    """兼容服务器迁移后 JSON 中仍保存旧 Windows 路径的情况。"""
    candidates = [Path(image_path)]
    if not Path(image_path).is_absolute():
        candidates.append(data_dir / image_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    filename = _filename_from_any_path(image_path)
    return filename_index.get(filename)


def load_split_items(label_dir: Path, split: str) -> List[Dict]:
    label_file = label_dir / f"{split}_labels.json"
    if not label_file.exists():
        raise FileNotFoundError(f"标注文件不存在: {label_file}")
    with open(label_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_copy_name(error_type: str, probability: float, index: int, source_path: str) -> str:
    suffix = Path(source_path).suffix.lower() or ".jpg"
    stem = Path(_filename_from_any_path(source_path)).stem
    return f"{error_type}_{index:04d}_p{probability:.6f}_{stem}{suffix}"


def export_binary_errors(args) -> str:
    import torch
    from PIL import Image

    from src.data_augmentation import get_val_transforms
    from src.ice_classifier import load_model

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["fn", "fp", "tp", "tn"]:
        if args.export_correct or subdir in {"fn", "fp"}:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    search_roots = [
        data_dir,
        data_dir / "data" / "roboflow_train",
        data_dir / "data" / "imagine",
        Path("data/roboflow_train"),
        Path("data/imagine"),
    ]
    filename_index = _build_filename_index(search_roots)
    items = load_split_items(label_dir, args.split)

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_classes=1,
        device=device,
    )
    model.eval()
    transform = get_val_transforms(image_size=args.image_size)

    rows = []
    missing = []
    image_batch = []
    meta_batch = []

    def flush_batch():
        if not image_batch:
            return

        images = torch.stack(image_batch).to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(images)).detach().cpu().numpy().reshape(-1)

        for meta, probability in zip(meta_batch, probabilities):
            true_label = meta["true_label"]
            error_type = classify_binary_error(true_label, float(probability), args.threshold)
            pred_label = int(float(probability) >= args.threshold)
            copied_path = ""

            should_copy = args.export_correct or error_type in {"FN", "FP"}
            if should_copy:
                subdir = output_dir / error_type.lower()
                copy_name = _safe_copy_name(error_type, float(probability), len(rows) + 1, meta["resolved_path"])
                dst = subdir / copy_name
                shutil.copy2(meta["resolved_path"], dst)
                copied_path = str(dst)

            rows.append({
                "image_path": meta["original_path"],
                "resolved_path": meta["resolved_path"],
                "true_label": true_label,
                "pred_prob": f"{float(probability):.8f}",
                "threshold": args.threshold,
                "pred_label": pred_label,
                "error_type": error_type,
                "copied_path": copied_path,
                "source": meta.get("source", ""),
            })

        image_batch.clear()
        meta_batch.clear()

    for item in items:
        original_path = item.get("image_path", "")
        resolved_path = resolve_image_path(original_path, data_dir, filename_index)
        if not resolved_path:
            missing.append(original_path)
            continue

        label_vector = item.get("label_vector", [0])
        true_label = int(bool(label_vector[0]))
        image = Image.open(resolved_path).convert("RGB")
        image_batch.append(transform(image))
        meta_batch.append({
            "original_path": original_path,
            "resolved_path": resolved_path,
            "true_label": true_label,
            "source": item.get("source", ""),
        })
        if len(image_batch) >= args.batch_size:
            flush_batch()

    flush_batch()

    csv_path = output_dir / "error_samples.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "image_path",
            "resolved_path",
            "true_label",
            "pred_prob",
            "threshold",
            "pred_label",
            "error_type",
            "copied_path",
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = {name: 0 for name in ["TP", "FP", "FN", "TN"]}
    for row in rows:
        stats[row["error_type"]] += 1

    print("\n=== 错误样本导出完成 ===")
    print(f"split: {args.split}")
    print(f"threshold: {args.threshold}")
    print(f"输出目录: {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"TP={stats['TP']}, FP={stats['FP']}, FN={stats['FN']}, TN={stats['TN']}")
    if missing:
        print(f"警告: 有 {len(missing)} 张图片找不到，已跳过。示例: {missing[:5]}")
    return str(csv_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="导出二分类模型的FN/FP错误样本")
    parser.add_argument("--checkpoint", type=str, default="weights/ice_binary_classifier/best_stage2.pth")
    parser.add_argument("--label-dir", type=str, default="data/binary_dataset")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/error_analysis_t0005")
    parser.add_argument("--export-correct", action="store_true", help="同时导出TP/TN样本；默认只导出FN/FP")
    args = parser.parse_args(argv)
    export_binary_errors(args)


if __name__ == "__main__":
    main()
