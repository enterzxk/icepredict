"""Prepare private-first 5-fold CV data for binary ice/snow training.

The private manual CSV is the only validation source. Roboflow data is kept as
auxiliary training data and never appears in validation folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BINARY_LABEL = "\u51b0\u96ea\u5f02\u5e38"
BINARY_LABEL_NAMES = [BINARY_LABEL]
MANUAL_LABELS = ["\u8986\u51b0", "\u96ea", "\u79ef\u96ea", "\u971c\u51bb"]
CSV_ENCODINGS = ["utf-8-sig", "gbk", "gb18030"]
ERROR_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?:FN|FP|TP|TN)_\d{4}", re.IGNORECASE)


def parse_label_value(value) -> int:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "\u5426", "\u65e0"}:
        return 0
    if text in {"1", "true", "yes", "y", "\u662f", "\u6709"}:
        return 1
    raise ValueError(f"label value must be 0/1, got {value!r}")


def read_csv_rows(csv_path: str) -> Tuple[List[Dict[str, str]], str]:
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in CSV_ENCODINGS:
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f)), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "manual_csv",
        b"",
        0,
        1,
        f"cannot read {csv_path} with {CSV_ENCODINGS}: {last_error}",
    )


def filename_from_any_path(image_path: str) -> str:
    return PureWindowsPath(str(image_path)).name or Path(str(image_path)).name


def resolve_manual_path(image_path: str, image_root: str = "") -> str:
    if not image_root:
        return image_path
    candidate = Path(image_path)
    if candidate.exists():
        return image_path
    root_candidate = Path(image_root) / filename_from_any_path(image_path)
    if root_candidate.exists():
        return str(root_candidate)
    return image_path


def read_manual_items(manual_csv: str, image_root: str = "") -> Tuple[List[Dict], Dict]:
    rows, encoding = read_csv_rows(manual_csv)
    if not rows:
        raise ValueError(f"manual csv is empty: {manual_csv}")

    missing_columns = [col for col in ["image_path", *MANUAL_LABELS] if col not in rows[0]]
    if missing_columns:
        raise ValueError(f"manual csv missing columns: {missing_columns}")

    items: List[Dict] = []
    per_label = {label: 0 for label in MANUAL_LABELS}
    errors: List[str] = []

    for line_no, row in enumerate(rows, start=2):
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            errors.append(f"line {line_no}: image_path is empty")
            continue
        try:
            multi_vector = [parse_label_value(row.get(label, "0")) for label in MANUAL_LABELS]
        except ValueError as exc:
            errors.append(f"line {line_no}: {exc}")
            continue

        binary_value = int(any(multi_vector))
        for label, value in zip(MANUAL_LABELS, multi_vector):
            per_label[label] += value

        items.append({
            "image_path": resolve_manual_path(image_path, image_root),
            "labels": {BINARY_LABEL: bool(binary_value)},
            "label_vector": [binary_value],
            "label_names": BINARY_LABEL_NAMES,
            "source": "private_manual_csv",
            "original_labels": {
                label: bool(value) for label, value in zip(MANUAL_LABELS, multi_vector)
            },
            "note": row.get("\u5907\u6ce8", ""),
        })

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"manual csv has {len(errors)} invalid rows:\n{preview}")

    positive = sum(int(item["label_vector"][0]) for item in items)
    summary = {
        "manual_csv": manual_csv,
        "encoding": encoding,
        "count": len(items),
        "positive": positive,
        "negative": len(items) - positive,
        "per_label": per_label,
    }
    return items, summary


def load_json_items(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"label file must be a list json: {path}")
    return data


def append_source(item: Dict, source: str) -> None:
    existing = item.get("source")
    if isinstance(existing, list):
        values = existing
    elif existing:
        values = [str(existing)]
    else:
        values = []
    values.append(source)
    item["source"] = list(dict.fromkeys(values))


def ensure_binary_item(raw_item: Dict, source: str) -> Dict:
    item = dict(raw_item)
    vector = item.get("label_vector", [0])
    value = int(bool(vector[0] if vector else 0))
    item["label_vector"] = [value]
    item["labels"] = {BINARY_LABEL: bool(value)}
    item["label_names"] = BINARY_LABEL_NAMES
    append_source(item, source)
    return item


def normalize_error_id(value: str) -> str:
    match = ERROR_ID_RE.search(str(value).strip())
    if not match:
        raise ValueError(f"invalid error id: {value!r}")
    return match.group(0).upper()


def read_wrong_ids(path: str) -> List[str]:
    if not path:
        return []
    ids: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            for match in ERROR_ID_RE.finditer(text):
                ids.append(match.group(0).upper())
    return list(dict.fromkeys(ids))


def extract_error_id(row: Dict[str, str]) -> Optional[str]:
    for field in ("copied_path", "image_path", "resolved_path"):
        value = row.get(field, "")
        match = ERROR_ID_RE.search(value)
        if match:
            return match.group(0).upper()
    return None


def load_error_rows(error_csv: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with open(error_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            error_id = extract_error_id(row)
            if error_id:
                rows[error_id] = row
    return rows


def path_keys(path: str) -> List[str]:
    text = str(path or "").strip()
    if not text:
        return []
    normalized = text.replace("\\", "/")
    keys = [
        normalized,
        os.path.normpath(text),
        filename_from_any_path(text),
    ]
    return list(dict.fromkeys(keys))


def build_item_index(items: Iterable[Dict]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        for key in path_keys(item.get("image_path", "")):
            index.setdefault(key, []).append(idx)
    return index


def find_item_index(error_row: Dict[str, str], item_index: Dict[str, List[int]]) -> Optional[int]:
    for field in ("image_path", "resolved_path"):
        for key in path_keys(error_row.get(field, "")):
            matches = item_index.get(key)
            if matches:
                return matches[0]
    return None


def apply_hard_negative_corrections(
    roboflow_items: List[Dict],
    error_csv: str = "",
    wrong_fn_ids: str = "",
) -> Dict:
    wrong_ids = read_wrong_ids(wrong_fn_ids) if wrong_fn_ids else []
    if not wrong_ids:
        return {
            "requested_ids": [],
            "corrected_ids": [],
            "missing_ids": [],
            "unmatched_ids": [],
            "warning": "",
        }

    if not error_csv or not Path(error_csv).exists():
        return {
            "requested_ids": wrong_ids,
            "corrected_ids": [],
            "missing_ids": wrong_ids,
            "unmatched_ids": [],
            "warning": "error_csv is missing, wrong ids were not applied",
        }

    error_rows = load_error_rows(error_csv)
    item_index = build_item_index(roboflow_items)
    corrected_ids: List[str] = []
    missing_ids: List[str] = []
    unmatched_ids: List[str] = []

    for error_id in wrong_ids:
        row = error_rows.get(error_id)
        if row is None:
            missing_ids.append(error_id)
            continue
        idx = find_item_index(row, item_index)
        if idx is None:
            unmatched_ids.append(error_id)
            continue

        item = roboflow_items[idx]
        item["label_vector"] = [0]
        item["labels"] = {BINARY_LABEL: False}
        item["label_names"] = BINARY_LABEL_NAMES
        item["hard_negative"] = True
        item["correction"] = {
            "error_id": error_id,
            "reason": "manual_wrong_fn_non_ice_or_uncertain",
        }
        append_source(item, "manual_hard_negative")
        corrected_ids.append(error_id)

    warning = ""
    if missing_ids or unmatched_ids:
        warning = "some wrong ids were not applied"
    return {
        "requested_ids": wrong_ids,
        "corrected_ids": corrected_ids,
        "missing_ids": missing_ids,
        "unmatched_ids": unmatched_ids,
        "warning": warning,
    }


def split_by_label(items: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    positives = [item for item in items if int(bool(item.get("label_vector", [0])[0])) == 1]
    negatives = [item for item in items if int(bool(item.get("label_vector", [0])[0])) == 0]
    return positives, negatives


def maybe_limit_items(items: List[Dict], max_count: int, seed: int) -> List[Dict]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    return copied[:max_count]


def make_stratified_folds(
    positives: Sequence[Dict],
    negatives: Sequence[Dict],
    fold_count: int,
    seed: int,
) -> List[List[Dict]]:
    if fold_count < 2:
        raise ValueError("folds must be >= 2")
    if len(positives) < fold_count:
        raise ValueError(f"positive private samples ({len(positives)}) < folds ({fold_count})")
    if len(negatives) < fold_count:
        raise ValueError(f"negative private samples ({len(negatives)}) < folds ({fold_count})")

    rng = random.Random(seed)
    pos = list(positives)
    neg = list(negatives)
    rng.shuffle(pos)
    rng.shuffle(neg)

    folds = [[] for _ in range(fold_count)]
    for idx, item in enumerate(pos):
        folds[idx % fold_count].append(item)
    for idx, item in enumerate(neg):
        folds[idx % fold_count].append(item)
    return folds


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_items(items: Sequence[Dict]) -> Dict[str, int]:
    positive = sum(int(bool(item.get("label_vector", [0])[0])) for item in items)
    return {
        "count": len(items),
        "positive": positive,
        "negative": len(items) - positive,
    }


def prepare_private_cv(args) -> Dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    private_items, private_summary = read_manual_items(args.manual_csv, args.image_root)
    private_pos, private_neg = split_by_label(private_items)

    roboflow_items = [
        ensure_binary_item(item, "roboflow_aux")
        for item in load_json_items(args.roboflow_labels)
    ]
    correction_summary = apply_hard_negative_corrections(
        roboflow_items=roboflow_items,
        error_csv=args.error_csv,
        wrong_fn_ids=args.wrong_fn_ids,
    )
    roboflow_pos, roboflow_neg = split_by_label(roboflow_items)
    hard_negative_items = [
        item for item in roboflow_neg
        if item.get("hard_negative") or "manual_hard_negative" in item.get("source", [])
    ]

    roboflow_pos = maybe_limit_items(roboflow_pos, args.max_roboflow_pos, args.seed + 11)
    roboflow_neg = maybe_limit_items(roboflow_neg, args.max_roboflow_neg, args.seed + 17)

    folds = make_stratified_folds(private_pos, private_neg, args.folds, args.seed)

    dump_json(output_dir / "private_labels.json", private_items)
    dump_json(output_dir / "roboflow_pos_labels.json", roboflow_pos)
    dump_json(output_dir / "roboflow_neg_labels.json", roboflow_neg)
    dump_json(output_dir / "hard_negative_labels.json", hard_negative_items)

    fold_infos = []
    for fold_idx, val_items in enumerate(folds):
        val_names = {item["image_path"] for item in val_items}
        train_private = [item for item in private_items if item["image_path"] not in val_names]
        train_labels = train_private + roboflow_pos + roboflow_neg

        fold_dir = output_dir / f"fold_{fold_idx}"
        dump_json(fold_dir / "train_private_labels.json", train_private)
        dump_json(fold_dir / "val_labels.json", val_items)
        dump_json(fold_dir / "train_roboflow_pos_labels.json", roboflow_pos)
        dump_json(fold_dir / "train_roboflow_neg_labels.json", roboflow_neg)
        dump_json(fold_dir / "train_hard_negative_labels.json", hard_negative_items)
        dump_json(fold_dir / "train_labels.json", train_labels)

        fold_info = {
            "fold": fold_idx,
            "train_private": count_items(train_private),
            "validation_private": count_items(val_items),
            "roboflow_pos": count_items(roboflow_pos),
            "roboflow_neg": count_items(roboflow_neg),
            "hard_negative": count_items(hard_negative_items),
            "validation_sources": sorted(set(str(item.get("source", "")) for item in val_items)),
        }
        dump_json(fold_dir / "fold_info.json", fold_info)
        fold_infos.append(fold_info)

    summary = {
        "output_dir": str(output_dir),
        "folds": args.folds,
        "seed": args.seed,
        "private": private_summary,
        "private_positive": len(private_pos),
        "private_negative": len(private_neg),
        "roboflow_positive": len(roboflow_pos),
        "roboflow_negative": len(roboflow_neg),
        "hard_negative": len(hard_negative_items),
        "corrections": correction_summary,
        "folds_detail": fold_infos,
    }
    dump_json(output_dir / "dataset_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare private-first binary CV folds")
    parser.add_argument("--manual-csv", default="manual_labels.csv")
    parser.add_argument("--image-root", default="", help="Optional image root for basename remap")
    parser.add_argument("--roboflow-labels", default="data/labels/binary_training_labels.json")
    parser.add_argument("--error-csv", default="")
    parser.add_argument("--wrong-fn-ids", default="")
    parser.add_argument("--output-dir", default="data/private_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-roboflow-pos", type=int, default=0)
    parser.add_argument("--max-roboflow-neg", type=int, default=0)
    args = parser.parse_args(argv)

    summary = prepare_private_cv(args)
    print("\n=== private CV data prepared ===")
    print(f"output_dir: {summary['output_dir']}")
    print(f"private: {summary['private_positive']} positive, {summary['private_negative']} negative")
    print(f"roboflow_aux: {summary['roboflow_positive']} positive, {summary['roboflow_negative']} negative")
    print(f"hard_negative: {summary['hard_negative']}")
    corrections = summary["corrections"]
    print(f"corrections_applied: {len(corrections['corrected_ids'])}/{len(corrections['requested_ids'])}")
    if corrections.get("warning"):
        print(f"warning: {corrections['warning']}")
        if corrections.get("missing_ids"):
            print(f"missing_ids: {corrections['missing_ids'][:10]}")
        if corrections.get("unmatched_ids"):
            print(f"unmatched_ids: {corrections['unmatched_ids'][:10]}")
    for fold in summary["folds_detail"]:
        val = fold["validation_private"]
        print(f"fold_{fold['fold']}: val pos={val['positive']}, neg={val['negative']}")


if __name__ == "__main__":
    main()
