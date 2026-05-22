"""
根据错误样本编号修正 Roboflow 测试集标签。

用途：把人工确认的错标 FN 样本从冰雪异常修正为正常样本，不删除原图。
"""

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Tuple


BINARY_LABEL_NAMES = ["冰雪异常"]
ERROR_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?:FN|FP|TP|TN)_\d{4}", re.IGNORECASE)


def normalize_error_id(value: str) -> str:
    match = ERROR_ID_RE.search(value.strip())
    if not match:
        raise ValueError(f"错误样本编号格式不正确: {value!r}")
    return match.group(0).upper()


def read_wrong_ids(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            for part in re.split(r"[\s,，;；]+", text):
                if part:
                    ids.append(normalize_error_id(part))
    return list(dict.fromkeys(ids))


def _path_basename(path: str) -> str:
    return PureWindowsPath(path).name or Path(path).name


def _path_keys(path: str) -> List[str]:
    text = str(path or "").strip()
    if not text:
        return []
    normalized = text.replace("\\", "/")
    keys = [normalized, os.path.normpath(text), _path_basename(text)]
    return list(dict.fromkeys(keys))


def extract_error_id(row: Dict[str, str]) -> Optional[str]:
    for field in ("copied_path", "image_path", "resolved_path"):
        value = row.get(field, "")
        match = ERROR_ID_RE.search(value)
        if match:
            return match.group(0).upper()
    return None


def load_error_rows(error_csv: str) -> Dict[str, Dict[str, str]]:
    with open(error_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            error_id = extract_error_id(row)
            if error_id:
                rows[error_id] = row
    return rows


def build_item_index(items: Iterable[Dict]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        for key in _path_keys(item.get("image_path", "")):
            index.setdefault(key, []).append(idx)
    return index


def find_item_index(row: Dict[str, str], item_index: Dict[str, List[int]]) -> Optional[int]:
    for field in ("image_path", "resolved_path"):
        for key in _path_keys(row.get(field, "")):
            matches = item_index.get(key)
            if matches:
                return matches[0]
    return None


def recompute_split_info(split_info: Dict, split_name: str, items: List[Dict]) -> Dict:
    fixed = dict(split_info)
    fixed.setdefault(split_name, {})
    fixed[split_name]["count"] = len(items)
    fixed[split_name]["files"] = [item.get("image_path", "") for item in items]
    positive = sum(int(bool(item.get("label_vector", [0])[0])) for item in items)
    fixed[split_name]["label_stats"] = {
        "冰雪异常": positive,
        "无": len(items) - positive,
    }
    return fixed


def apply_corrections(
    label_dir: str,
    error_csv: str,
    wrong_fn_ids: str,
    output_dir: str,
) -> Dict[str, object]:
    label_root = Path(label_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    wrong_ids = read_wrong_ids(wrong_fn_ids)
    error_rows = load_error_rows(error_csv)
    missing_ids = [error_id for error_id in wrong_ids if error_id not in error_rows]
    if missing_ids:
        raise ValueError(f"这些编号没有在 error_samples.csv 中找到: {missing_ids}")

    for split in ("train", "val"):
        src = label_root / f"{split}_labels.json"
        dst = output_root / f"{split}_labels.json"
        if src.exists():
            shutil.copy2(src, dst)

    test_file = label_root / "test_labels.json"
    if not test_file.exists():
        raise FileNotFoundError(f"测试集标签不存在: {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_items = json.load(f)

    item_index = build_item_index(test_items)
    corrected = []
    unmatched = []
    for error_id in wrong_ids:
        row = error_rows[error_id]
        idx = find_item_index(row, item_index)
        if idx is None:
            unmatched.append(error_id)
            continue
        item = test_items[idx]
        original_vector = list(item.get("label_vector", [0]))
        item["label_vector"] = [0]
        item["labels"] = {"冰雪异常": False}
        item["label_names"] = BINARY_LABEL_NAMES
        item["correction"] = {
            "error_id": error_id,
            "original_label_vector": original_vector,
            "reason": "wrong_fn_non_ice_or_uncertain",
        }
        corrected.append(error_id)

    if unmatched:
        raise ValueError(f"这些编号找到了错误行，但没有匹配到 test_labels.json: {unmatched}")

    with open(output_root / "test_labels.json", "w", encoding="utf-8") as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)

    split_info_file = label_root / "split_info.json"
    if split_info_file.exists():
        with open(split_info_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        split_info = recompute_split_info(split_info, "test", test_items)
        with open(output_root / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

    summary = {
        "input_label_dir": str(label_root),
        "output_dir": str(output_root),
        "requested_ids": wrong_ids,
        "corrected_ids": corrected,
        "corrected_count": len(corrected),
    }
    with open(output_root / "correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="按 FN_xxxx 编号修正二分类测试集标签")
    parser.add_argument("--label-dir", required=True, help="原始划分目录，例如 data/binary_dataset")
    parser.add_argument("--error-csv", required=True, help="error_samples.csv 路径")
    parser.add_argument("--wrong-fn-ids", required=True, help="一行一个 FN_xxxx 的文本文件")
    parser.add_argument("--output-dir", required=True, help="修正后划分输出目录")
    args = parser.parse_args(argv)

    summary = apply_corrections(
        label_dir=args.label_dir,
        error_csv=args.error_csv,
        wrong_fn_ids=args.wrong_fn_ids,
        output_dir=args.output_dir,
    )
    print("\n=== 测试集标签修正完成 ===")
    print(f"输入目录: {summary['input_label_dir']}")
    print(f"输出目录: {summary['output_dir']}")
    print(f"修正样本数: {summary['corrected_count']}")
    print(f"修正编号: {', '.join(summary['corrected_ids'])}")


if __name__ == "__main__":
    main()
