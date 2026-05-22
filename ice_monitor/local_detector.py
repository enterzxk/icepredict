"""
本地图像检测器
封装本地图型推理接口，支持多标签输出
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ice_classifier import load_model
from src.data_augmentation import get_inference_transforms


# 标签名称
LABEL_NAMES = ["覆冰", "雪", "积雪", "霜冻"]


class LocalIceDetector:
    """
    本地覆冰图像检测器

    使用本地训练的ResNet50模型进行多标签分类

    Args:
        checkpoint_path: 模型checkpoint路径
        model_name: 模型名称
        device: 推理设备
        threshold: 分类阈值
        image_size: 图像尺寸
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        model_name: str = "resnet50",
        device: str = None,
        threshold: float = 0.5,
        image_size: int = 224,
    ):
        # 设备
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 默认checkpoint路径
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                _PROJECT_ROOT, "weights", "ice_classifier", "best_stage2.pth"
            )

        # 加载模型
        self.model = load_model(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            num_classes=4,
            device=self.device,
        )
        self.model.eval()

        # 数据变换
        self.transform = get_inference_transforms(image_size=image_size)

        # 配置
        self.threshold = threshold
        self.image_size = image_size

        print(f"[Local Detector] 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  阈值: {self.threshold}")
        print(f"  图像尺寸: {self.image_size}")

    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载并预处理图像"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image.unsqueeze(0)  # 添加batch维度
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            return torch.zeros(1, 3, self.image_size, self.image_size)

    def detect(self, image_path: str) -> Dict:
        """
        检测单张图像

        Args:
            image_path: 图像路径

        Returns:
            dict: {
                'image_path': str,
                'label': str,           # 最可能的标签组合
                'confidence': float,    # 置信度
                'details': dict,        # 每个类别的概率
                'labels': list,         # 预测的标签列表
            }
        """
        # 加载图像
        image_tensor = self._load_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # 获取预测结果
        predictions = (probabilities > self.threshold).astype(int)
        predicted_labels = []
        details = {}

        for i, label_name in enumerate(LABEL_NAMES):
            details[label_name] = float(probabilities[i])
            if predictions[i] == 1:
                predicted_labels.append(label_name)

        # 构建标签字符串
        if predicted_labels:
            label_str = "+".join(predicted_labels)
            confidence = float(np.mean([probabilities[i] for i, p in enumerate(predictions) if p == 1]))
        else:
            label_str = "无覆冰"
            confidence = float(1.0 - np.max(probabilities))

        return {
            "image_path": image_path,
            "label": label_str,
            "confidence": confidence,
            "details": details,
            "labels": predicted_labels,
        }

    def detect_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        批量检测图像

        Args:
            image_paths: 图像路径列表

        Returns:
            list of dicts
        """
        results = []
        for image_path in image_paths:
            result = self.detect(image_path)
            results.append(result)
        return results

    def detect_folder(
        self,
        image_dir: str,
        output: str = "local_detect_results.json",
    ) -> List[Dict]:
        """
        批量检测图像目录

        Args:
            image_dir: 图像目录路径
            output: 结果JSON文件路径

        Returns:
            list of dicts
        """
        import json

        # 查找图像
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
        image_paths = sorted(
            str(f.resolve())
            for f in Path(image_dir).rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        )

        print(f"[Local Detector] 共找到 {len(image_paths)} 张图片")

        # 批量检测
        results = []
        for idx, path in enumerate(image_paths, 1):
            print(f"  [{idx}/{len(image_paths)}] {os.path.basename(path)}")
            result = self.detect(path)
            results.append(result)

            # 实时保存
            with open(output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[Local Detector] 完成，结果已保存至: {output}")
        return results

    def to_vlm_format(self, result: Dict) -> Dict:
        """
        将本地检测结果转换为VLM格式（兼容原有接口）

        Args:
            result: 本地检测结果

        Returns:
            dict: VLM格式结果
        """
        # 将多标签结果转换为yes/no/unknow
        labels = result.get("labels", [])
        if labels:
            # 如果有任何覆冰相关标签，返回yes
            label = "yes"
        else:
            label = "no"

        return {
            "image_path": result["image_path"],
            "label": label,
            "raw_response": str(result),
        }


def create_detector(
    checkpoint_path: str = None,
    device: str = None,
    threshold: float = 0.5,
) -> LocalIceDetector:
    """
    创建检测器的便捷函数

    Args:
        checkpoint_path: 模型checkpoint路径
        device: 推理设备
        threshold: 分类阈值

    Returns:
        LocalIceDetector
    """
    return LocalIceDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
    )
