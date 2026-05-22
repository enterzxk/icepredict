"""
将人工标注 CSV 转成二分类验证集 JSON。

规则：覆冰/雪/积雪/霜冻任一为 1 => 冰雪异常；四列全 0 => 正常。
"""

import argparse
import csv
import json
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple


LABEL_NAMES = ["覆冰", "雪", "积雪", "霜冻"]
BINARY_LABEL_NAMES = ["冰雪异常"]
CSV_ENCODINGS = ["utf-8-sig", "gbk", "gb18030"]


def parse_label_value(value) -> int:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "否", "无"}:
        return 0
    if text in {"1", "true", "yes", "y", "是", "有"}:
        return 1
    raise ValueError(f"标签值只能是 0/1，当前值: {value!r}")


def read_manual_csv(csv_path: str) -> Tuple[List[Dict], str]:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            return rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "manual_csv",
        b"",
        0,
        1,
        f"无法用 {CSV_ENCODINGS} 读取 {csv_path}: {last_error}",
    )


def _basename(path: str) -> str:
    return PureWindowsPath(path).name or Path(path).name


def resolve_manual_path(image_path: str, image_root: str = "") -> str:
    if not image_root:
        return image_path
    candidate = Path(image_path)
    if candidate.exists():
        return image_path
    root_candidate = Path(image_root) / _basename(image_path)
    if root_candidate.exists():
        return str(root_candidate)
    return image_path


def convert_manual_csv(
    manual_csv: str,
    output_dir: str,
    image_root: str = "",
) -> Dict[str, object]:
    rows, encoding = read_manual_csv(manual_csv)
    if not rows:
        raise ValueError(f"人工标注CSV为空: {manual_csv}")

    missing_columns = [col for col in ["image_path", *LABEL_NAMES] if col not in rows[0]]
    if missing_columns:
        raise ValueError(f"CSV缺少列: {missing_columns}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    data = []
    positive = 0
    per_label = {label: 0 for label in LABEL_NAMES}
    errors = []
    for line_no, row in enumerate(rows, start=2):
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            errors.append(f"第{line_no}行 image_path 为空")
            continue
        try:
            multi_vector = [parse_label_value(row.get(label, "0")) for label in LABEL_NAMES]
        except ValueError as exc:
            errors.append(f"第{line_no}行 {exc}")
            continue

        binary_value = int(any(multi_vector))
        positive += binary_value
        for idx, label in enumerate(LABEL_NAMES):
            per_label[label] += multi_vector[idx]

        data.append({
            "image_path": resolve_manual_path(image_path, image_root),
            "labels": {"冰雪异常": bool(binary_value)},
            "label_vector": [binary_value],
            "label_names": BINARY_LABEL_NAMES,
            "source": "manual_csv_binary",
            "original_labels": {
                label: bool(multi_vector[idx])
                for idx, label in enumerate(LABEL_NAMES)
            },
            "note": row.get("备注", ""),
        })

    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"CSV存在 {len(errors)} 个问题，请先修正。\n{preview}")

    for name in ["manual_labels_binary.json", "test_labels.json"]:
        with open(output_root / name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    summary = {
        "manual_csv": manual_csv,
        "encoding": encoding,
        "output_dir": str(output_root),
        "total": len(data),
        "positive": positive,
        "negative": len(data) - positive,
        "per_label": per_label,
    }
    with open(output_root / "manual_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="将人工CSV转为冰雪二分类验证集")
    parser.add_argument("--manual-csv", required=True, help="人工标注CSV")
    parser.add_argument("--output-dir", default="data/manual_eval", help="输出目录")
    parser.add_argument("--image-root", default="", help="可选：按文件名重映射到该图片目录")
    args = parser.parse_args(argv)

    summary = convert_manual_csv(args.manual_csv, args.output_dir, args.image_root)
    print("\n=== 人工验证集转换完成 ===")
    print(f"CSV编码: {summary['encoding']}")
    print(f"样本数: {summary['total']}")
    print(f"冰雪异常: {summary['positive']}")
    print(f"正常: {summary['negative']}")
    print(f"四标签统计: {summary['per_label']}")
    print(f"输出目录: {summary['output_dir']}")


if __name__ == "__main__":
    main()
