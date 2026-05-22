"""Generate local patch candidates from private monitoring images.

This does not change source images. It creates cropped patch images and a CSV
template for manual patch-level labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_private_cv import MANUAL_LABELS, parse_label_value, read_csv_rows  # noqa: E402


BINARY_LABEL = "\u51b0\u96ea\u5f02\u5e38"
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
CSV_FIELDS = [
    "patch_path",
    "patch_label",
    "quality",
    "source_image",
    "source_binary_label",
    "x",
    "y",
    "w",
    "h",
    "rank_score",
    "reason",
    "original_labels",
    "note",
]


def filename_from_any_path(image_path: str) -> str:
    return PureWindowsPath(str(image_path)).name or Path(str(image_path)).name


def resolve_image_path(image_path: str, image_root: str = "") -> Optional[Path]:
    candidates = [Path(image_path)]
    if image_root:
        candidates.append(Path(image_root) / filename_from_any_path(image_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_manual_rows(manual_csv: str) -> Tuple[List[Dict], Dict]:
    rows, encoding = read_csv_rows(manual_csv)
    if not rows:
        raise ValueError(f"manual csv is empty: {manual_csv}")
    missing = [name for name in ["image_path", *MANUAL_LABELS] if name not in rows[0]]
    if missing:
        raise ValueError(f"manual csv missing columns: {missing}")

    parsed = []
    per_label = {name: 0 for name in MANUAL_LABELS}
    for line_no, row in enumerate(rows, start=2):
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            raise ValueError(f"line {line_no}: image_path is empty")
        vector = [parse_label_value(row.get(name, "0")) for name in MANUAL_LABELS]
        for name, value in zip(MANUAL_LABELS, vector):
            per_label[name] += value
        parsed.append({
            "image_path": image_path,
            "binary_label": int(any(vector)),
            "original_labels": {
                name: int(value) for name, value in zip(MANUAL_LABELS, vector)
            },
            "note": row.get("\u5907\u6ce8", ""),
        })

    positive = sum(item["binary_label"] for item in parsed)
    return parsed, {
        "encoding": encoding,
        "total": len(parsed),
        "positive": positive,
        "negative": len(parsed) - positive,
        "per_label": per_label,
    }


def axis_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_windows(width: int, height: int, patch_size: int, stride: int) -> List[Tuple[int, int, int, int]]:
    xs = axis_positions(width, patch_size, stride)
    ys = axis_positions(height, patch_size, stride)
    windows = []
    for y in ys:
        for x in xs:
            right = min(width, x + patch_size)
            lower = min(height, y + patch_size)
            windows.append((x, y, right - x, lower - y))
    return windows


def add_center_windows(width: int, height: int, sizes: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    windows = []
    for size in sizes:
        crop_w = min(width, size)
        crop_h = min(height, size)
        x = max(0, (width - crop_w) // 2)
        y = max(0, (height - crop_h) // 2)
        windows.append((x, y, crop_w, crop_h))
    return windows


def crop_rank_score(image: Image.Image, box: Tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    crop = image.crop((x, y, x + w, y + h)).convert("L")
    crop.thumbnail((96, 96))
    stat = ImageStat.Stat(crop)
    mean = stat.mean[0]
    stddev = stat.stddev[0]
    hist = crop.histogram()
    total = max(1, sum(hist))
    bright = sum(hist[210:]) / total
    midbright = sum(hist[160:]) / total
    return float(mean * 0.15 + stddev * 1.2 + bright * 120.0 + midbright * 35.0)


def dedupe_windows(windows: Sequence[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    result = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        seen.add(window)
        result.append(window)
    return result


def select_windows(
    image: Image.Image,
    binary_label: int,
    patch_size: int,
    stride: int,
    max_positive_patches: int,
    max_negative_patches: int,
    seed: int,
) -> List[Tuple[Tuple[int, int, int, int], float, str]]:
    width, height = image.size
    windows = make_windows(width, height, patch_size, stride)
    windows.extend(add_center_windows(width, height, [patch_size, int(patch_size * 1.35)]))
    windows = dedupe_windows(windows)

    scored = [(window, crop_rank_score(image, window)) for window in windows]
    scored.sort(key=lambda item: item[1], reverse=True)

    if binary_label:
        limit = max_positive_patches
        selected = scored[:limit]
        reason = "private_positive_ranked"
    else:
        limit = max_negative_patches
        rng = random.Random(seed)
        top_count = max(1, math.ceil(limit * 0.6))
        top = scored[:top_count]
        rest = scored[top_count:]
        rng.shuffle(rest)
        selected = (top + rest)[:limit]
        reason = "private_negative_hard_and_random"

    return [(window, score, reason) for window, score in selected]


def save_patch(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    output_path: Path,
    output_size: int,
    jpeg_quality: int,
) -> None:
    x, y, w, h = box
    patch = image.crop((x, y, x + w, y + h))
    if output_size > 0 and patch.size != (output_size, output_size):
        patch = patch.resize((output_size, output_size), RESAMPLE_BICUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(output_path, format="JPEG", quality=jpeg_quality)


def make_patch_candidates(args) -> Dict:
    rows, manual_summary = read_manual_rows(args.manual_csv)
    if args.limit_images > 0:
        rows = rows[:args.limit_images]
    output_dir = Path(args.output_dir)
    patch_dir = output_dir / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Dict] = []
    missing = []
    positive_patch_count = 0
    negative_patch_count = 0

    for image_index, row in enumerate(rows):
        resolved = resolve_image_path(row["image_path"], args.image_root)
        if not resolved:
            missing.append(row["image_path"])
            continue

        try:
            image = Image.open(resolved).convert("RGB")
        except Exception as exc:
            missing.append(f"{row['image_path']} ({exc})")
            continue

        selected = select_windows(
            image=image,
            binary_label=row["binary_label"],
            patch_size=args.patch_size,
            stride=args.stride,
            max_positive_patches=args.max_positive_patches,
            max_negative_patches=args.max_negative_patches,
            seed=args.seed + image_index,
        )

        label_folder = "positive_source" if row["binary_label"] else "negative_source"
        stem = Path(filename_from_any_path(row["image_path"])).stem
        for patch_index, (box, score, reason) in enumerate(selected):
            patch_name = f"{image_index:04d}_{patch_index:03d}_{stem}.jpg"
            patch_path = patch_dir / label_folder / patch_name
            save_patch(image, box, patch_path, args.output_size, args.jpeg_quality)

            x, y, w, h = box
            if row["binary_label"]:
                positive_patch_count += 1
            else:
                negative_patch_count += 1
            csv_rows.append({
                "patch_path": str(patch_path).replace("\\", "/"),
                "patch_label": "",
                "quality": "",
                "source_image": row["image_path"],
                "source_binary_label": row["binary_label"],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "rank_score": f"{score:.4f}",
                "reason": reason,
                "original_labels": json.dumps(row["original_labels"], ensure_ascii=False),
                "note": row["note"],
            })

    csv_path = output_dir / "patch_candidates.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "manual_csv": args.manual_csv,
        "image_root": args.image_root,
        "output_dir": str(output_dir),
        "patch_dir": str(patch_dir),
        "csv_path": str(csv_path),
        "manual_summary": manual_summary,
        "patch_count": len(csv_rows),
        "positive_source_patch_count": positive_patch_count,
        "negative_source_patch_count": negative_patch_count,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "config": vars(args),
    }
    with open(output_dir / "patch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate patch candidates for manual frost/ice labeling")
    parser.add_argument("--manual-csv", default="manual_labels.csv")
    parser.add_argument("--image-root", default="data/imagine")
    parser.add_argument("--output-dir", default="data/patch_candidates")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--output-size", type=int, default=384)
    parser.add_argument("--max-positive-patches", type=int, default=16)
    parser.add_argument("--max-negative-patches", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-images", type=int, default=0, help="Only process first N images for smoke tests")
    args = parser.parse_args(argv)

    summary = make_patch_candidates(args)
    print("\n=== Patch candidates generated ===")
    print(f"manual images: {summary['manual_summary']['total']}")
    print(f"manual positives: {summary['manual_summary']['positive']}")
    print(f"manual negatives: {summary['manual_summary']['negative']}")
    print(f"patches: {summary['patch_count']}")
    print(f"positive-source patches: {summary['positive_source_patch_count']}")
    print(f"negative-source patches: {summary['negative_source_patch_count']}")
    print(f"missing images: {summary['missing_count']}")
    print(f"CSV: {summary['csv_path']}")
    print("Fill patch_label with 1/0, use quality=uncertain for unclear patches.")


if __name__ == "__main__":
    main()
