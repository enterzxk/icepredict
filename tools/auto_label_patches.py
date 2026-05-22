"""Auto-label patch_candidates.csv with a conservative vision model prompt."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_ENCODINGS = ["utf-8-sig", "gbk", "gb18030"]
EXTRA_FIELDS = ["vlm_label", "vlm_quality", "vlm_confidence", "vlm_reason", "vlm_raw_response"]

PROMPT = """
You are labeling cropped monitoring-image patches from power transmission equipment.

Task: decide whether this patch visibly contains ice/snow/frost abnormality.

Return ONLY valid JSON:
{
  "patch_label": 0 or 1 or null,
  "quality": "ok" or "uncertain",
  "confidence": number from 0 to 1,
  "reason": "short Chinese reason"
}

Strict conservative rules:
- Use patch_label=1 only when visible frozen material is present on power equipment, line, insulator, tower part, fitting, or nearby vegetation/equipment: frost texture, rime, ice coating, icicle, snow accumulation, or obvious frozen white crystalline deposit.
- Thin/light frost is positive if it is visibly attached to equipment or line, even if not severe.
- Use patch_label=0 when the patch is normal equipment/line/background, rain, fog, haze, lens water, glare, white sky, overexposure, metal reflection, paint, dust, or cloud without visible frozen deposit on an object.
- Use patch_label=null and quality="uncertain" when the patch is too dark, too blurry, heavily occluded, mostly blank/background, no relevant object is visible, or frost cannot be distinguished from glare/fog/overexposure.
- Do not infer ice/snow only from weather, gray tone, low temperature text, fog, or camera timestamp.
- Prefer false negatives over false positives. If unsure, mark uncertain.
""".strip()


def read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str], str]:
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader), list(reader.fieldnames or []), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError("patch_csv", b"", 0, 1, f"cannot read {path}: {last_error}")


def write_csv_atomic(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def encode_image_base64(image_path: str) -> Tuple[str, str]:
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    media_type = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return image_data, media_type


def parse_json_response(text: str) -> Dict:
    cleaned = str(text or "").strip()
    if not cleaned:
        return make_uncertain_result("模型未返回文本内容", text)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return make_uncertain_result("模型未返回JSON，已按不确定处理", text)

    json_text = match.group(0)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return make_uncertain_result("模型返回的JSON无法解析，已按不确定处理", text)

    label = data.get("patch_label")
    if label is None or str(label).strip().lower() in {"null", "none", ""}:
        normalized_label = None
    else:
        normalized_label = int(bool(int(label)))

    quality = str(data.get("quality", "ok")).strip().lower()
    if quality not in {"ok", "uncertain"}:
        quality = "uncertain"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "patch_label": normalized_label,
        "quality": quality,
        "confidence": confidence,
        "reason": str(data.get("reason", "")).strip(),
        "raw": text,
    }


def make_uncertain_result(reason: str, raw_text: str = "") -> Dict:
    return {
        "patch_label": None,
        "quality": "uncertain",
        "confidence": 0.0,
        "reason": reason,
        "raw": str(raw_text or ""),
    }


def extract_anthropic_text(response) -> str:
    """Extract text from Anthropic-style responses that may include thinking blocks."""
    texts = []
    block_types = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", type(block).__name__)
        block_types.append(str(block_type))
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))
    if texts:
        return "\n".join(texts).strip()
    return f"Anthropic response has no text block, content block types={block_types}"


def extract_openai_text(message) -> str:
    """Extract text from OpenAI-compatible message content variants."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()

    texts = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
            if text:
                texts.append(str(text))
    if texts:
        return "\n".join(texts).strip()

    refusal = getattr(message, "refusal", None)
    if refusal:
        return str(refusal).strip()
    return ""


class PatchVisionLabeler:
    def __init__(
        self,
        api_format: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float,
    ):
        self.api_format = api_format
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        if not api_key:
            raise ValueError("missing API key, set env var or pass --api-key")

        if api_format == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)
        elif api_format == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            raise ValueError(f"unsupported api_format: {api_format}")

    def label(self, image_path: str) -> Dict:
        image_data, media_type = encode_image_base64(image_path)
        if self.api_format == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                    ],
                }],
            )
            text = extract_anthropic_text(response)
        else:
            data_url = f"data:{media_type};base64,{image_data}"
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
            )
            text = extract_openai_text(response.choices[0].message)
        return parse_json_response(text)


def row_is_done(row: Dict[str, str]) -> bool:
    patch_label = str(row.get("patch_label", "")).strip()
    quality = str(row.get("quality", "")).strip().lower()
    return bool(patch_label) or quality in {"uncertain", "不确定", "看不清", "模糊", "夜间"}


def apply_result_to_row(row: Dict[str, str], result: Dict) -> None:
    label = result["patch_label"]
    quality = result["quality"]
    confidence = result["confidence"]

    if label is None or quality == "uncertain":
        row["patch_label"] = ""
        row["quality"] = "uncertain"
    else:
        row["patch_label"] = str(label)
        row["quality"] = "ok"

    row["vlm_label"] = "" if label is None else str(label)
    row["vlm_quality"] = quality
    row["vlm_confidence"] = f"{confidence:.4f}"
    row["vlm_reason"] = result.get("reason", "")
    row["vlm_raw_response"] = result.get("raw", "")


def auto_label_patches(args) -> Dict:
    rows, fieldnames, encoding = read_csv_rows(args.patch_csv)
    if "patch_path" not in fieldnames:
        raise ValueError("patch csv must contain patch_path column")
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    output_csv = args.output_csv or args.patch_csv
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    labeler = None
    if not args.dry_run:
        labeler = PatchVisionLabeler(
            api_format=args.api_format,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.timeout,
        )

    processed = 0
    skipped = 0
    failed = 0

    for idx, row in enumerate(rows):
        if args.max_items > 0 and processed >= args.max_items:
            break
        if not args.overwrite and row_is_done(row):
            skipped += 1
            continue

        patch_path = str(row.get("patch_path", "")).strip()
        if not patch_path:
            failed += 1
            row["quality"] = "uncertain"
            row["vlm_reason"] = "patch_path为空"
            continue
        if not Path(patch_path).exists():
            failed += 1
            row["quality"] = "uncertain"
            row["vlm_reason"] = f"patch文件不存在: {patch_path}"
            continue

        print(f"[{idx + 1}/{len(rows)}] {patch_path}")
        if args.dry_run:
            result = {
                "patch_label": None,
                "quality": "uncertain",
                "confidence": 0.0,
                "reason": "dry_run",
                "raw": "{}",
            }
            apply_result_to_row(row, result)
            processed += 1
            continue

        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                assert labeler is not None
                result = labeler.label(patch_path)
                apply_result_to_row(row, result)
                processed += 1
                print(
                    f"  label={row.get('patch_label') or 'uncertain'} "
                    f"quality={row.get('quality')} conf={row.get('vlm_confidence')} "
                    f"reason={row.get('vlm_reason')}"
                )
                break
            except Exception as exc:
                last_error = exc
                print(f"  retry {attempt}/{args.retries}: {exc}")
                time.sleep(args.retry_sleep * attempt)
        else:
            failed += 1
            row["patch_label"] = ""
            row["quality"] = "uncertain"
            row["vlm_label"] = ""
            row["vlm_quality"] = "uncertain"
            row["vlm_confidence"] = "0.0000"
            row["vlm_reason"] = f"VLM失败: {last_error}"

        if args.save_every > 0 and (processed + failed) % args.save_every == 0:
            write_csv_atomic(output_csv, rows, fieldnames)
            print(f"  saved: {output_csv}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_csv_atomic(output_csv, rows, fieldnames)

    labeled_pos = sum(1 for row in rows if str(row.get("patch_label", "")).strip() == "1")
    labeled_neg = sum(1 for row in rows if str(row.get("patch_label", "")).strip() == "0")
    uncertain = sum(1 for row in rows if str(row.get("quality", "")).strip().lower() == "uncertain")
    summary = {
        "patch_csv": args.patch_csv,
        "output_csv": output_csv,
        "encoding": encoding,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "positive": labeled_pos,
        "negative": labeled_neg,
        "uncertain": uncertain,
        "config": vars(args),
    }
    summary_path = Path(output_csv).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auto-label patch_candidates.csv with a conservative VLM")
    parser.add_argument("--patch-csv", default="data/patch_candidates/patch_candidates.csv")
    parser.add_argument("--output-csv", default="", help="Default: overwrite patch-csv atomically")
    parser.add_argument("--api-format", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = auto_label_patches(args)
    print("\n=== Patch VLM auto-label finished ===")
    print(f"processed: {summary['processed']}")
    print(f"skipped: {summary['skipped']}")
    print(f"failed: {summary['failed']}")
    print(f"positive: {summary['positive']}")
    print(f"negative: {summary['negative']}")
    print(f"uncertain: {summary['uncertain']}")
    print(f"output_csv: {summary['output_csv']}")


if __name__ == "__main__":
    main()
