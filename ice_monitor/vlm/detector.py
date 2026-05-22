"""
ice_monitor.vlm — VLM零样本覆冰图像识别模块
封装自 snow_ice_detector_plus_v2
"""
import os
import json
import time
import base64
from pathlib import Path
from typing import List, Optional

# =========== 配置 ===========
_MODEL_NAME = "qwen-vl-plus"   # 可替换为 qwen3.5-flash / qwen-vl-max
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

_PROMPT = """
你是一个图像判别助手。
请判断这张图片中是否存在"覆冰"、"雪"、"积雪"或"霜冻"。

要求：
1. 只判断图片中是否真的出现覆冰、雪、积雪、霜冻等视觉内容。图片中可能出现圆柱形玻璃绝缘子，注意与覆冰区分。
2. 有覆冰、雪、积雪或霜冻，返回 yes。
3. 没有覆冰及霜冻，返回 no。
4. 如果不确定，返回 unknow。
5. 只输出 JSON，不输出其他解释。

输出格式严格如下：
{"label": "yes"} 或 {"label": "no"} 或 {"label": "unknow"}
""".strip()


def _encode_image(image_path: str) -> str:
    """将图像编码为 base64 data URL"""
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


def _parse_label(text: str) -> str:
    """解析 VLM 输出 JSON，返回 yes/no/unknow"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        label = json.loads(text).get("label", "").strip().lower()
        return label if label in {"yes", "no"} else "unknow"
    except Exception:
        return "unknow"


def find_images(folder: str) -> List[str]:
    """递归查找目录下所有图像文件"""
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    return sorted(
        str(f.resolve()) for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    )


class IceImageDetector:
    """
    VLM 零样本覆冰图像识别器

    使用方法：
        detector = IceImageDetector(api_key="sk-xxx")

        # 识别单张图片
        result = detector.detect("path/to/image.jpg")

        # 批量识别整个目录，支持断点续传
        detector.detect_folder("images/", output="results.json")

    Args:
        api_key (str): 阿里云 DashScope API Key（也可通过环境变量 DASHSCOPE_API_KEY 设置）
        model (str):   VLM模型名称，默认 qwen-vl-plus
        max_retries (int): 单张图片最大重试次数
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _MODEL_NAME,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库：pip install openai")

        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise ValueError(
                "缺少 API Key。请传入 api_key 参数，或设置环境变量 DASHSCOPE_API_KEY"
            )

        self.client = OpenAI(api_key=key, base_url=_BASE_URL)
        self.model = model
        self.max_retries = max_retries
        print(f"[VLM Detector] 初始化完成 | 模型: {self.model}")

    def detect(self, image_path: str) -> dict:
        """
        识别单张图像

        Returns:
            dict: {
                'image_path': str,
                'label': str,          # 'yes' / 'no' / 'unknow' / 'error'
                'raw_response': str,
            }
        """
        data_url = _encode_image(image_path)
        last_err = None

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    }],
                    temperature=0,
                )
                content = resp.choices[0].message.content
                return {
                    "image_path": image_path,
                    "label": _parse_label(content),
                    "raw_response": content,
                }
            except Exception as e:
                last_err = str(e)
                print(f"  ⚠ 重试 {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2)

        return {"image_path": image_path, "label": "error", "raw_response": None, "error": last_err}

    def detect_folder(self, image_dir: str, output: str = "snow_ice_result.json") -> List[dict]:
        """
        批量识别图像目录，支持断点续传

        Args:
            image_dir (str): 图像目录路径
            output (str):    结果 JSON 文件路径

        Returns:
            list of dicts，每项同 detect() 的返回格式
        """
        all_images = find_images(image_dir)
        print(f"[VLM Detector] 共找到 {len(all_images)} 张图片")

        # 读取已有结果（断点续传）
        results = []
        if os.path.exists(output):
            try:
                with open(output, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                results = []

        done = {
            item["image_path"] for item in results
            if item.get("label") in {"yes", "no"}
        }
        remaining = [p for p in all_images if p not in done]
        print(f"[VLM Detector] 已完成 {len(done)} 张，本次处理 {len(remaining)} 张")

        for idx, path in enumerate(remaining, 1):
            print(f"  [{idx}/{len(remaining)}] {os.path.basename(path)}")
            result = self.detect(path)
            results = [r for r in results if r.get("image_path") != path]
            results.append(result)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[VLM Detector] 完成，结果已保存至: {output}")
        return results

    @staticmethod
    def generate_label_template(image_dir: str, output: str = "human_labels.json"):
        """生成人工标注模板文件"""
        images = find_images(image_dir)
        template = [{"image_path": p, "label": ""} for p in images]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"[VLM Detector] 标注模板已生成: {output}（{len(template)} 张图片）")
        print("  请在每条记录的 label 字段填入 'yes' 或 'no'")
