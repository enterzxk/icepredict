"""
自动标注工具 - 使用VLM对图像进行多标签标注
识别目标：覆冰、雪、积雪、霜冻
支持小米MiMo（Anthropic格式）和阿里云DashScope（OpenAI格式）
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ice_monitor.vlm.detector import find_images

LABEL_NAMES = ["覆冰", "雪", "积雪", "霜冻"]
BINARY_LABEL_NAMES = ["冰雪异常"]
MANUAL_CSV_COLUMNS = ["image_path", *LABEL_NAMES, "备注"]


# 多标签标注提示词
_MULTI_LABEL_PROMPT = """
你是一个输电线路冰雪图像判别助手。请先判断图像是否“可判读”，再判断是否存在以下现象：

1. **覆冰**：输电线路、金具、绝缘子或设备表面有明确冰层、冰挂、冰柱、冰壳等覆冰现象
2. **雪**：图像中存在明确雪景、降雪或地面/背景积雪
3. **积雪**：设备、线路、横担、绝缘子等目标物上有明确积雪覆盖
4. **霜冻**：设备、线路或金具表面有明确霜、结晶状白霜、霜花

必须严格遵守：
1. 夜间黑暗、红外强曝光、画面过曝/欠曝导致目标细节不可见时，标为“不确定”，四个标签全部为 false。
2. 镜头严重模糊、失焦、抖动、雨滴/水雾/污渍遮挡镜头、画面大面积白雾或看不清线路设备时，标为“不确定”，四个标签全部为 false。
3. 只是下雨、雾天、湿润、镜头水珠、灰白雾气、反光、曝光发白，不等于雪、积雪、霜冻或覆冰；没有明确固态冰雪证据时不要标 true。
4. 只有在能清楚看到“固态冰雪/霜冻”附着在目标或场景中时，才把对应标签设为 true。
5. 图片中可能出现圆柱形玻璃绝缘子、亮白金属、反光、电缆纹理，不要误判为覆冰。
6. 对四个类别独立判断。一张图可以有多个 true。
7. 只输出 JSON，不输出其他解释。

输出格式严格如下：
{
  "不确定": true/false,
  "不确定原因": "若不确定，简短说明原因；否则为空字符串",
  "覆冰": true/false,
  "雪": true/false,
  "积雪": true/false,
  "霜冻": true/false
}
""".strip()


class MultiLabelAutoLabeler:
    """
    多标签自动标注器

    使用VLM对图像进行多标签标注，识别覆冰、雪、积雪、霜冻四种现象
    支持小米MiMo（Anthropic格式）和阿里云DashScope（OpenAI格式）

    Args:
        api_key: API Key
        model: 模型名称
        base_url: API地址
        api_format: API格式，"anthropic" 或 "openai"
        max_retries: 单张图片最大重试次数
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mimo-v2.5",
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        api_format: str = "openai",
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url
        self.api_format = api_format
        self.max_retries = max_retries

        # 获取API Key（优先使用参数，其次环境变量）
        key = api_key or os.environ.get("MIMO_API_KEY", "")
        if not key:
            raise ValueError(
                "缺少 API Key。请传入 api_key 参数，或设置环境变量 MIMO_API_KEY"
            )

        # 根据API格式初始化客户端
        if api_format == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=key, base_url=base_url)
            except ImportError:
                raise ImportError("请安装 anthropic 库：pip install anthropic")
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=key, base_url=base_url)
            except ImportError:
                raise ImportError("请安装 openai 库：pip install openai")

        print(f"[Auto Labeler] 初始化完成")
        print(f"  模型: {self.model}")
        print(f"  API格式: {self.api_format}")
        print(f"  API地址: {self.base_url}")

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 data URL"""
        import base64
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp", ".gif": "image/gif",
        }
        mime = mime_map.get(suffix, "application/octet-stream")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _parse_labels(self, text: str) -> Dict[str, bool]:
        """解析VLM输出的JSON，返回多标签结果"""
        import json

        def to_bool(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "1", "有", "是"}
            return False

        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").replace("json", "", 1).strip()

        try:
            result = json.loads(text)
            # 确保所有标签都是布尔值
            labels = {}
            for key in ["覆冰", "雪", "积雪", "霜冻"]:
                labels[key] = to_bool(result.get(key, False))
            uncertain = to_bool(result.get("不确定", False))
            if uncertain:
                labels = {key: False for key in ["覆冰", "雪", "积雪", "霜冻"]}
            return {
                "labels": labels,
                "uncertain": uncertain,
                "uncertain_reason": str(result.get("不确定原因", "") or ""),
            }
        except Exception:
            # 解析失败时不进入训练集，避免把坏响应当成负样本
            return {
                "labels": {"覆冰": False, "雪": False, "积雪": False, "霜冻": False},
                "uncertain": True,
                "uncertain_reason": "模型输出无法解析",
            }

    def label_single(self, image_path: str) -> Dict:
        """
        标注单张图像

        Returns:
            dict: {
                'image_path': str,
                'labels': {'覆冰': bool, '雪': bool, '积雪': bool, '霜冻': bool},
                'raw_response': str,
            }
        """
        import time
        import base64

        # 读取图像并编码
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # 获取MIME类型
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp", ".gif": "image/gif",
        }
        media_type = mime_map.get(suffix, "image/jpeg")

        last_err = None

        for attempt in range(self.max_retries):
            try:
                if self.api_format == "anthropic":
                    # Anthropic API格式
                    resp = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": _MULTI_LABEL_PROMPT,
                                },
                            ],
                        }],
                    )
                    content = resp.content[0].text
                else:
                    # OpenAI API格式
                    data_url = f"data:{media_type};base64,{image_data}"
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _MULTI_LABEL_PROMPT},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ]
                        }],
                        temperature=0,
                    )
                    content = resp.choices[0].message.content

                parsed = self._parse_labels(content)
                return {
                    "image_path": image_path,
                    "labels": parsed["labels"],
                    "uncertain": parsed["uncertain"],
                    "uncertain_reason": parsed["uncertain_reason"],
                    "raw_response": content,
                }
            except Exception as e:
                last_err = str(e)
                print(f"  ⚠ 重试 {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2)

        return {
            "image_path": image_path,
            "labels": {"覆冰": False, "雪": False, "积雪": False, "霜冻": False},
            "uncertain": True,
            "uncertain_reason": "API请求失败",
            "raw_response": None,
            "error": last_err,
        }

    def label_folder(
        self,
        image_dir: str,
        output: str = "multi_label_results.json",
        resume: bool = True,
        max_consecutive_errors: int = 5,
    ) -> List[Dict]:
        """
        批量标注图像目录，支持断点续传

        Args:
            image_dir: 图像目录路径
            output: 结果JSON文件路径
            resume: 是否断点续传
            max_consecutive_errors: 连续失败多少张后停止，避免网络故障时污染结果

        Returns:
            list of dicts
        """
        all_images = find_images(image_dir)
        print(f"[Auto Labeler] 共找到 {len(all_images)} 张图片")

        # 读取已有结果（断点续传）
        results = []
        if resume and os.path.exists(output):
            try:
                with open(output, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                results = []

        done = {
            item["image_path"]
            for item in results
            if "labels" in item and not item.get("error")
        }
        remaining = [p for p in all_images if p not in done]
        print(f"[Auto Labeler] 已完成 {len(done)} 张，本次处理 {len(remaining)} 张")

        consecutive_errors = 0
        for idx, path in enumerate(remaining, 1):
            print(f"  [{idx}/{len(remaining)}] {os.path.basename(path)}")
            result = self.label_single(path)
            results = [r for r in results if r.get("image_path") != path]
            results.append(result)

            # 实时保存
            with open(output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if result.get("error"):
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"连续 {consecutive_errors} 张图片标注失败，已暂停。"
                        "请检查网络、代理、API Key或API地址后重新运行；失败图片不会被视为已完成。"
                    )
            else:
                consecutive_errors = 0

        print(f"[Auto Labeler] 完成，结果已保存至: {output}")
        return results


def convert_to_training_format(
    label_file: str,
    output_dir: str = "data/labels",
    threshold: float = 0.5,
    task: str = "multi_label",
    output_name: Optional[str] = None,
):
    """
    将VLM标注结果转换为训练所需格式

    Args:
        label_file: VLM标注结果JSON文件
        output_dir: 输出目录
        threshold: 置信度阈值（用于过滤不确定样本）
    """
    with open(label_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    if task not in {"multi_label", "binary"}:
        raise ValueError(f"不支持的任务类型: {task}")

    # 统计标签分布
    stats = {"覆冰": 0, "雪": 0, "积雪": 0, "霜冻": 0, "无": 0, "不确定": 0}
    training_data = []
    source_count = len(results)

    for item in results:
        if "labels" not in item or item.get("error"):
            continue
        if item.get("uncertain"):
            stats["不确定"] += 1
            continue

        labels = item["labels"]
        has_any = any(labels.values())

        # 统计
        for key, val in labels.items():
            if val:
                stats[key] += 1
        if not has_any:
            stats["无"] += 1

        multi_label_vector = [
            int(labels.get("覆冰", False)),
            int(labels.get("雪", False)),
            int(labels.get("积雪", False)),
            int(labels.get("霜冻", False)),
        ]
        if task == "binary":
            binary_value = int(any(multi_label_vector))
            training_data.append({
                "image_path": item["image_path"],
                "labels": {"冰雪异常": bool(binary_value)},
                "label_vector": [binary_value],
                "label_names": BINARY_LABEL_NAMES,
                "source": "vlm_binary",
            })
        else:
            # 转换为训练格式
            training_data.append({
                "image_path": item["image_path"],
                "labels": labels,
                "label_vector": multi_label_vector,
                "label_names": LABEL_NAMES,
                "source": "vlm_multi_label",
            })

    # 保存训练格式
    if output_name is None:
        output_name = "binary_training_labels.json" if task == "binary" else "training_labels.json"
    output_file = os.path.join(output_dir, output_name)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n=== 标签分布统计 ===")
    print(f"总标注数: {source_count}")
    print(f"有效训练样本数: {len(training_data)}")
    for key, count in stats.items():
        ratio = count / source_count * 100 if source_count else 0
        print(f"{key}: {count} ({ratio:.1f}%)")
    print(f"\n训练标签已保存至: {output_file}")
    return output_file


def _parse_manual_label(value) -> int:
    """解析人工标注CSV中的0/1值。空值默认按0处理。"""
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "否", "无"}:
        return 0
    if text in {"1", "true", "yes", "y", "是", "有"}:
        return 1
    raise ValueError(f"标签值只能填0/1，当前值: {value!r}")


def make_manual_template(
    image_dir: str = "data/imagine",
    output: str = "manual_labels.csv",
) -> str:
    """
    生成人工标注CSV模板；如果文件已存在，会保留已有标注并补充新增图片。
    """
    image_paths = find_images(image_dir)
    existing_rows = {}

    if os.path.exists(output):
        with open(output, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("image_path", "").strip()
                if path:
                    existing_rows[path] = row

    rows = []
    for image_path in image_paths:
        row = existing_rows.get(image_path, {})
        rows.append({
            "image_path": image_path,
            "覆冰": row.get("覆冰", "0"),
            "雪": row.get("雪", "0"),
            "积雪": row.get("积雪", "0"),
            "霜冻": row.get("霜冻", "0"),
            "备注": row.get("备注", ""),
        })

    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANUAL_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"人工标注模板已生成: {output}")
    print(f"图像数量: {len(rows)}")
    print("请在 覆冰/雪/积雪/霜冻 四列填写 0 或 1，填完后运行 --convert-manual-csv")
    return output


def convert_manual_csv_to_training_format(
    csv_file: str = "manual_labels.csv",
    output_dir: str = "data/labels",
    task: str = "multi_label",
    output_name: Optional[str] = None,
) -> str:
    """
    将人工标注CSV转换为训练所需的training_labels.json。
    CSV列: image_path,覆冰,雪,积雪,霜冻,备注
    """
    os.makedirs(output_dir, exist_ok=True)
    if task not in {"multi_label", "binary"}:
        raise ValueError(f"不支持的任务类型: {task}")

    training_data = []
    stats = {label: 0 for label in LABEL_NAMES}
    stats["无"] = 0
    errors = []

    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = [col for col in MANUAL_CSV_COLUMNS[:-1] if col not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"CSV缺少列: {', '.join(missing_columns)}")

        for line_no, row in enumerate(reader, start=2):
            image_path = row.get("image_path", "").strip()
            if not image_path:
                errors.append(f"第{line_no}行: image_path为空")
                continue
            if not os.path.exists(image_path):
                errors.append(f"第{line_no}行: 图片不存在: {image_path}")
                continue

            try:
                label_vector = [_parse_manual_label(row.get(label, "0")) for label in LABEL_NAMES]
            except ValueError as exc:
                errors.append(f"第{line_no}行: {exc}")
                continue

            labels = {
                label: bool(label_vector[idx])
                for idx, label in enumerate(LABEL_NAMES)
            }

            if any(label_vector):
                for idx, label in enumerate(LABEL_NAMES):
                    stats[label] += label_vector[idx]
            else:
                stats["无"] += 1

            if task == "binary":
                binary_value = int(any(label_vector))
                training_data.append({
                    "image_path": image_path,
                    "labels": {"冰雪异常": bool(binary_value)},
                    "label_vector": [binary_value],
                    "label_names": BINARY_LABEL_NAMES,
                    "source": "manual_csv_binary",
                    "original_labels": labels,
                    "note": row.get("备注", ""),
                })
            else:
                training_data.append({
                    "image_path": image_path,
                    "labels": labels,
                    "label_vector": label_vector,
                    "label_names": LABEL_NAMES,
                    "source": "manual_csv",
                    "note": row.get("备注", ""),
                })

    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(
            f"人工标注CSV存在 {len(errors)} 个问题，请先修正。\n{preview}"
        )

    if output_name is None:
        output_name = "binary_manual_labels.json" if task == "binary" else "training_labels.json"
    output_file = os.path.join(output_dir, output_name)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print("\n=== 人工标注分布统计 ===")
    print(f"总样本数: {len(training_data)}")
    for key, count in stats.items():
        ratio = count / len(training_data) * 100 if training_data else 0
        print(f"{key}: {count} ({ratio:.1f}%)")
    print(f"\n训练标签已保存至: {output_file}")
    return output_file


def _parse_keyword_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def convert_coco_to_binary_training_format(
    coco_file: str,
    image_dir: Optional[str] = None,
    output_dir: str = "data/labels",
    output_name: str = "binary_roboflow_labels.json",
    positive_category_keywords: Optional[str] = None,
    include_unannotated_negative: bool = False,
) -> str:
    """
    将COCO检测标注转换成二分类训练JSON。

    默认只把类别名包含 ice accretion / ice coating / snow / frost / icing 的框视为正样本；
    例如 Roboflow 数据中的 objects on power lines 不会被当作冰雪异常。
    """
    os.makedirs(output_dir, exist_ok=True)
    coco_path = Path(coco_file)
    if image_dir is None:
        image_root = coco_path.parent
    else:
        image_root = Path(image_dir)

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    keywords = _parse_keyword_list(positive_category_keywords)
    if not keywords:
        keywords = ["ice accretion", "ice coating", "snow", "frost", "icing"]

    categories = coco.get("categories", [])
    category_names = {
        category.get("id"): str(category.get("name", ""))
        for category in categories
    }
    positive_category_ids = {
        category_id
        for category_id, category_name in category_names.items()
        if any(keyword in category_name.lower() for keyword in keywords)
    }
    if not positive_category_ids:
        raise ValueError(
            "没有找到可作为正样本的COCO类别。"
            f"当前类别: {list(category_names.values())}; 关键词: {keywords}"
        )

    annotations_by_image = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann.get("image_id"), []).append(ann)

    training_data = []
    missing_images = []
    positive_count = 0
    negative_count = 0

    for image in coco.get("images", []):
        image_id = image.get("id")
        annotations = annotations_by_image.get(image_id, [])
        if not annotations and not include_unannotated_negative:
            continue

        image_path = image_root / image.get("file_name", "")
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue

        category_ids = [ann.get("category_id") for ann in annotations]
        category_labels = [category_names.get(category_id, str(category_id)) for category_id in category_ids]
        is_positive = any(category_id in positive_category_ids for category_id in category_ids)
        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

        training_data.append({
            "image_path": str(image_path),
            "labels": {"冰雪异常": is_positive},
            "label_vector": [int(is_positive)],
            "label_names": BINARY_LABEL_NAMES,
            "source": "roboflow_coco_binary",
            "coco_categories": sorted(set(category_labels)),
        })

    output_file = os.path.join(output_dir, output_name)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print("\n=== COCO二分类转换统计 ===")
    print(f"COCO图片数: {len(coco.get('images', []))}")
    print(f"输出样本数: {len(training_data)}")
    print(f"正样本(冰雪异常): {positive_count}")
    print(f"负样本(非冰雪/其他缺陷): {negative_count}")
    print(f"正样本类别关键词: {keywords}")
    print(f"正样本COCO类别: {[category_names[i] for i in sorted(positive_category_ids)]}")
    if missing_images:
        print(f"警告: 有 {len(missing_images)} 张图片在本地不存在，已跳过。示例: {missing_images[:3]}")
    print(f"\n训练标签已保存至: {output_file}")
    return output_file


def merge_training_label_files(input_files: List[str], output_file: str) -> str:
    """
    合并多个训练JSON。相同图片重复出现时采用逐位或逻辑，二分类下只要任一来源为正就保留为正。
    """
    merged = {}
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            image_path = item.get("image_path")
            if not image_path:
                continue
            key = os.path.abspath(image_path)
            vector = [int(bool(v)) for v in item.get("label_vector", [])]
            if not vector and item.get("labels"):
                label_names = item.get("label_names") or BINARY_LABEL_NAMES
                vector = [int(bool(item["labels"].get(name, False))) for name in label_names]

            if key not in merged:
                normalized = dict(item)
                normalized["image_path"] = image_path
                normalized["label_vector"] = vector
                normalized.setdefault("label_names", BINARY_LABEL_NAMES if len(vector) == 1 else LABEL_NAMES[:len(vector)])
                normalized["source"] = [str(item.get("source", input_file))]
                merged[key] = normalized
                continue

            old = merged[key]
            old_vector = [int(bool(v)) for v in old.get("label_vector", [])]
            max_len = max(len(old_vector), len(vector))
            old_vector += [0] * (max_len - len(old_vector))
            vector += [0] * (max_len - len(vector))
            new_vector = [int(a or b) for a, b in zip(old_vector, vector)]
            old["label_vector"] = new_vector
            label_names = old.get("label_names") or (BINARY_LABEL_NAMES if len(new_vector) == 1 else LABEL_NAMES[:len(new_vector)])
            old["labels"] = {
                name: bool(new_vector[idx])
                for idx, name in enumerate(label_names)
            }
            old.setdefault("source", [])
            if not isinstance(old["source"], list):
                old["source"] = [old["source"]]
            old["source"].append(str(item.get("source", input_file)))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_data = list(merged.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    positives = sum(1 for item in merged_data if any(item.get("label_vector", [])))
    print("\n=== 标签文件合并统计 ===")
    print(f"输入文件数: {len(input_files)}")
    print(f"合并后样本数: {len(merged_data)}")
    print(f"正样本数: {positives}")
    print(f"负样本数: {len(merged_data) - positives}")
    print(f"合并结果已保存至: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="VLM多标签自动标注工具")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/imagine",
        help="图像目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multi_label_results.json",
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mimo-v2.5",
        help="VLM模型名称"
    )
    parser.add_argument(
        "--api-format",
        type=str,
        choices=["anthropic", "openai"],
        default="openai",
        help="API格式"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://token-plan-cn.xiaomimimo.com/v1",
        help="API地址"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key；默认读取 MIMO_API_KEY"
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="连续失败多少张后停止"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="将标注结果转换为训练格式"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["multi_label", "binary"],
        default="multi_label",
        help="转换任务类型；binary 会把任意覆冰/雪/积雪/霜冻合并为冰雪异常"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="转换输出文件名；默认按任务自动选择"
    )
    parser.add_argument(
        "--make-manual-template",
        action="store_true",
        help="生成Excel/WPS可编辑的人工标注CSV模板"
    )
    parser.add_argument(
        "--manual-output",
        type=str,
        default="manual_labels.csv",
        help="人工标注CSV模板输出路径"
    )
    parser.add_argument(
        "--convert-manual-csv",
        action="store_true",
        help="将人工标注CSV转换为训练格式"
    )
    parser.add_argument(
        "--manual-csv",
        type=str,
        default="manual_labels.csv",
        help="人工标注CSV输入路径"
    )
    parser.add_argument(
        "--convert-input",
        type=str,
        default="multi_label_results.json",
        help="转换时的输入文件"
    )
    parser.add_argument(
        "--convert-output",
        type=str,
        default="data/labels",
        help="转换时的输出目录"
    )
    parser.add_argument(
        "--convert-coco-binary",
        action="store_true",
        help="将COCO检测标注转换为二分类训练格式"
    )
    parser.add_argument(
        "--coco-file",
        type=str,
        default=None,
        help="COCO标注文件路径，例如 C:/Users/京康/Desktop/train/_annotations.coco.json"
    )
    parser.add_argument(
        "--coco-image-dir",
        type=str,
        default=None,
        help="COCO图片所在目录；默认使用COCO文件所在目录"
    )
    parser.add_argument(
        "--positive-category-keywords",
        type=str,
        default="ice accretion,ice coating,snow,frost,icing",
        help="逗号分隔的正样本类别关键词"
    )
    parser.add_argument(
        "--include-unannotated-negative",
        action="store_true",
        help="COCO中无标注图片也作为负样本纳入"
    )
    parser.add_argument(
        "--merge-label-files",
        nargs="+",
        default=None,
        help="合并多个训练JSON文件"
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default="data/labels/binary_training_labels.json",
        help="合并输出文件路径"
    )
    args = parser.parse_args()

    if args.make_manual_template:
        make_manual_template(args.image_dir, args.manual_output)
    elif args.convert_manual_csv:
        convert_manual_csv_to_training_format(
            args.manual_csv,
            args.convert_output,
            task=args.task,
            output_name=args.output_name,
        )
    elif args.convert:
        convert_to_training_format(
            args.convert_input,
            args.convert_output,
            task=args.task,
            output_name=args.output_name,
        )
    elif args.convert_coco_binary:
        if not args.coco_file:
            raise ValueError("--convert-coco-binary 需要提供 --coco-file")
        convert_coco_to_binary_training_format(
            coco_file=args.coco_file,
            image_dir=args.coco_image_dir,
            output_dir=args.convert_output,
            output_name=args.output_name or "binary_roboflow_labels.json",
            positive_category_keywords=args.positive_category_keywords,
            include_unannotated_negative=args.include_unannotated_negative,
        )
    elif args.merge_label_files:
        merge_training_label_files(args.merge_label_files, args.merge_output)
    else:
        labeler = MultiLabelAutoLabeler(
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            api_format=args.api_format,
        )
        labeler.label_folder(
            args.image_dir,
            args.output,
            max_consecutive_errors=args.max_consecutive_errors,
        )


if __name__ == "__main__":
    main()
