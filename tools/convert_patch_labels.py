"""Convert manually labeled patch CSV to binary training JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


BINARY_LABEL = "\u51b0\u96ea\u5f02\u5e38"
BINARY_LABEL_NAMES = [BINARY_LABEL]
CSV_ENCODINGS = ["utf-8-sig", "gbk", "gb18030"]


def read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], str]:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f)), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError("patch_csv", b"", 0, 1, f"cannot read {path}: {last_error}")


def parse_patch_label(value: str) -> int:
    text = "" if value is None else str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "\u662f", "\u6709"}:
        return 1
    if text in {"0", "false", "no", "n", "\u5426", "\u65e0"}:
        return 0
    raise ValueError(f"patch_label must be 0/1, got {value!r}")


def is_uncertain(row: Dict[str, str]) -> bool:
    text = str(row.get("quality", "")).strip().lower()
    return text in {"uncertain", "bad", "blur", "dark", "\u4e0d\u786e\u5b9a", "\u770b\u4e0d\u6e05", "\u6a21\u7cca", "\u591c\u95f4"}


def split_train_val(items: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    positives = [item for item in items if int(item["label_vector"][0]) == 1]
    negatives = [item for item in items if int(item["label_vector"][0]) == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def take_val(group: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        if not group:
            return [], []
        count = max(1, int(round(len(group) * val_ratio))) if len(group) >= 5 else 1
        count = min(count, len(group))
        return group[count:], group[:count]

    train_pos, val_pos = take_val(positives)
    train_neg, val_neg = take_val(negatives)
    train = train_pos + train_neg
    val = val_pos + val_neg
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def convert_patch_labels(args) -> Dict:
    rows, encoding = read_csv_rows(args.patch_csv)
    if not rows:
        raise ValueError(f"patch csv is empty: {args.patch_csv}")
    if "patch_path" not in rows[0] or "patch_label" not in rows[0]:
        raise ValueError("patch csv must contain patch_path and patch_label columns")

    items: List[Dict] = []
    skipped_unlabeled = 0
    skipped_uncertain = 0
    missing_files = []
    errors = []

    for line_no, row in enumerate(rows, start=2):
        patch_path = str(row.get("patch_path", "")).strip()
        if not patch_path:
            errors.append(f"line {line_no}: patch_path is empty")
            continue
        if is_uncertain(row):
            skipped_uncertain += 1
            continue
        raw_label = str(row.get("patch_label", "")).strip()
        if raw_label == "":
            skipped_unlabeled += 1
            continue
        try:
            label = parse_patch_label(raw_label)
        except ValueError as exc:
            errors.append(f"line {line_no}: {exc}")
            continue
        if args.require_exists and not Path(patch_path).exists():
            missing_files.append(patch_path)
            continue

        items.append({
            "image_path": patch_path,
            "labels": {BINARY_LABEL: bool(label)},
            "label_vector": [label],
            "label_names": BINARY_LABEL_NAMES,
            "source": "manual_patch_csv",
            "source_image": row.get("source_image", ""),
            "crop_box": {
                "x": int(float(row.get("x", 0) or 0)),
                "y": int(float(row.get("y", 0) or 0)),
                "w": int(float(row.get("w", 0) or 0)),
                "h": int(float(row.get("h", 0) or 0)),
            },
            "source_binary_label": int(float(row.get("source_binary_label", 0) or 0)),
            "quality": row.get("quality", ""),
            "reason": row.get("reason", ""),
            "note": row.get("note", ""),
        })

    if errors:
        raise ValueError("invalid patch csv rows:\n" + "\n".join(errors[:20]))
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} patch files are missing, examples: {missing_files[:10]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_items, val_items = split_train_val(items, args.val_ratio, args.seed)

    for name, data in [
        ("patch_labels.json", items),
        ("train_labels.json", train_items),
        ("val_labels.json", val_items),
        ("test_labels.json", val_items),
    ]:
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    positive = sum(int(item["label_vector"][0]) for item in items)
    summary = {
        "patch_csv": args.patch_csv,
        "encoding": encoding,
        "output_dir": str(output_dir),
        "total_labeled": len(items),
        "positive": positive,
        "negative": len(items) - positive,
        "train": {
            "count": len(train_items),
            "positive": sum(int(item["label_vector"][0]) for item in train_items),
        },
        "val": {
            "count": len(val_items),
            "positive": sum(int(item["label_vector"][0]) for item in val_items),
        },
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_uncertain": skipped_uncertain,
        "config": vars(args),
    }
    summary["train"]["negative"] = summary["train"]["count"] - summary["train"]["positive"]
    summary["val"]["negative"] = summary["val"]["count"] - summary["val"]["positive"]
    with open(output_dir / "patch_label_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert patch_candidates.csv after manual patch labeling")
    parser.add_argument("--patch-csv", default="data/patch_candidates/patch_candidates.csv")
    parser.add_argument("--output-dir", default="data/patch_dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-exists", action="store_true")
    args = parser.parse_args(argv)

    summary = convert_patch_labels(args)
    print("\n=== Patch labels converted ===")
    print(f"labeled patches: {summary['total_labeled']}")
    print(f"positive: {summary['positive']}")
    print(f"negative: {summary['negative']}")
    print(f"train: {summary['train']}")
    print(f"val: {summary['val']}")
    print(f"skipped unlabeled: {summary['skipped_unlabeled']}")
    print(f"skipped uncertain: {summary['skipped_uncertain']}")
    print(f"output_dir: {summary['output_dir']}")


if __name__ == "__main__":
    main()
