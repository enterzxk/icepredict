"""
数据标注工具 - 小米MiMo版本
使用小米MiMo API进行多标签标注
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_api_config, get_api_key
from ice_monitor.vlm.detector import find_images

# 多标签标注提示词
_MULTI_LABEL_PROMPT = """
你是一个图像判别助手。请判断这张图片中是否存在以下现象：

1. **覆冰**：输电线路或设备上的覆冰现象
2. **雪**：图像中存在雪景或积雪
3. **积雪**：设备或线路上的积雪覆盖
4. **霜冻**：设备或线路上的霜冻现象

要求：
1. 仔细观察图片内容，区分不同的冰雪现象
2. 图片中可能出现圆柱形玻璃绝缘子，注意与覆冰区分
3. 对每个类别独立判断，返回 true 或 false
4. 只输出 JSON，不输出其他解释

输出格式严格如下：
{
  "覆冰": true/false,
  "雪": true/false,
  "积雪": true/false,
  "霜冻": true/false
}
""".strip()


class MiMoAutoLabeler:
    """
    小米MiMo自动标注器

    使用小米MiMo API对图像进行多标签标注

    Args:
        api_key: API Key（也可通过环境变量 MIMO_API_KEY 设置）
        model: 模型名称
        base_url: API地址
        max_retries: 单张图片最大重试次数
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        base_url: str = None,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库：pip install openai")

        # 获取配置
        config = get_api_config("mimo")

        self.api_key = api_key or get_api_key("mimo")
        self.model = model or config["model"]
        self.base_url = base_url or config["base_url"]
        self.max_retries = max_retries

        # 创建客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        print(f"[MiMo Auto Labeler] 初始化完成")
        print(f"  模型: {self.model}")
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
        """解析模型输出的JSON，返回多标签结果"""
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
            return labels
        except Exception:
            # 解析失败时返回全False
            return {"覆冰": False, "雪": False, "积雪": False, "霜冻": False}

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
        data_url = self._encode_image(image_path)
        last_err = None

        for attempt in range(self.max_retries):
            try:
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
                labels = self._parse_labels(content)
                return {
                    "image_path": image_path,
                    "labels": labels,
                    "raw_response": content,
                }
            except Exception as e:
                last_err = str(e)
                print(f"  ⚠ 重试 {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2)

        return {
            "image_path": image_path,
            "labels": {"覆冰": False, "雪": False, "积雪": False, "霜冻": False},
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
        print(f"[MiMo Auto Labeler] 共找到 {len(all_images)} 张图片")

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
        print(f"[MiMo Auto Labeler] 已完成 {len(done)} 张，本次处理 {len(remaining)} 张")

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

        print(f"[MiMo Auto Labeler] 完成，结果已保存至: {output}")
        return results


def convert_to_training_format(
    label_file: str,
    output_dir: str = "data/labels",
):
    """
    将标注结果转换为训练所需格式

    Args:
        label_file: 标注结果JSON文件
        output_dir: 输出目录
    """
    with open(label_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 统计标签分布
    stats = {"覆冰": 0, "雪": 0, "积雪": 0, "霜冻": 0, "无": 0}
    training_data = []

    for item in results:
        if "labels" not in item or item.get("error"):
            continue

        labels = item["labels"]
        has_any = any(labels.values())

        # 统计
        for key, val in labels.items():
            if val:
                stats[key] += 1
        if not has_any:
            stats["无"] += 1

        # 转换为训练格式
        training_data.append({
            "image_path": item["image_path"],
            "labels": labels,
            "label_vector": [
                int(labels.get("覆冰", False)),
                int(labels.get("雪", False)),
                int(labels.get("积雪", False)),
                int(labels.get("霜冻", False)),
            ]
        })

    # 保存训练格式
    output_file = os.path.join(output_dir, "training_labels.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n=== 标签分布统计 ===")
    print(f"总样本数: {len(training_data)}")
    for key, count in stats.items():
        ratio = count / len(training_data) * 100 if training_data else 0
        print(f"{key}: {count} ({ratio:.1f}%)")
    print(f"\n训练标签已保存至: {output_file}")


def main():
    # 配置参数
    IMAGE_DIR = "data/imagine"
    OUTPUT_FILE = "multi_label_results.json"

    print("="*60)
    print("小米MiMo自动标注工具")
    print("="*60)

    # 检查是否已有标注结果
    if os.path.exists(OUTPUT_FILE):
        print(f"标注结果文件已存在: {OUTPUT_FILE}")
        choice = input("是否重新标注? (y/n): ").strip().lower()
        if choice != 'y':
            print("跳过标注，转换为训练格式...")
            convert_to_training_format(OUTPUT_FILE, "data/labels")
            return

    # 运行标注
    print(f"\n开始标注图像目录: {IMAGE_DIR}")
    labeler = MiMoAutoLabeler()
    labeler.label_folder(IMAGE_DIR, OUTPUT_FILE)

    # 转换为训练格式
    print("\n转换为训练格式...")
    convert_to_training_format(OUTPUT_FILE, "data/labels")

    print("\n标注完成！")


if __name__ == "__main__":
    main()
