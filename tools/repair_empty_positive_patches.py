"""Repair positive-source patch rows that were marked uncertain due to empty VLM responses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


CSV_ENCODINGS = ["utf-8-sig", "gbk", "gb18030"]
VLM_FIELDS = ["vlm_label", "vlm_quality", "vlm_confidence", "vlm_reason", "vlm_raw_response"]


def read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str], str]:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader), list(reader.fieldnames or []), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError("patch_csv", b"", 0, 1, f"cannot read {path}: {last_error}")


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def should_repair(row: Dict[str, str], reason_contains: str) -> bool:
    if str(row.get("source_binary_label", "")).strip() != "1":
        return False
    if str(row.get("patch_label", "")).strip():
        return False
    if str(row.get("quality", "")).strip().lower() != "uncertain":
        return False
    return reason_contains in str(row.get("vlm_reason", ""))


def repair_empty_positive_patches(args) -> Dict[str, object]:
    rows, fieldnames, encoding = read_csv_rows(args.patch_csv)
    for field in VLM_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    repaired = 0
    for row in rows:
        if not should_repair(row, args.reason_contains):
            continue
        row["patch_label"] = "1"
        row["quality"] = "ok"
        row["vlm_label"] = "1"
        row["vlm_quality"] = "empty_response_positive_fallback"
        row["vlm_confidence"] = f"{args.confidence:.4f}"
        row["vlm_reason"] = "模型空回复；positive_source默认保留为正样本"
        repaired += 1

    output_csv = args.output_csv or args.patch_csv
    write_csv(output_csv, rows, fieldnames)
    summary = {
        "patch_csv": args.patch_csv,
        "output_csv": output_csv,
        "encoding": encoding,
        "repaired": repaired,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Repair empty-response uncertain positive-source patch labels")
    parser.add_argument("--patch-csv", default="data/patch_candidates/patch_candidates_mimo.csv")
    parser.add_argument("--output-csv", default="", help="Default: overwrite patch-csv")
    parser.add_argument("--reason-contains", default="模型未返回文本内容")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args(argv)

    summary = repair_empty_positive_patches(args)
    print("\n=== Empty positive patch repair finished ===")
    print(f"input: {summary['patch_csv']}")
    print(f"output: {summary['output_csv']}")
    print(f"repaired: {summary['repaired']}")


if __name__ == "__main__":
    main()
