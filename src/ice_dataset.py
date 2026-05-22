"""
覆冰图像数据集类
支持多标签分类：[覆冰, 雪, 积雪, 霜冻]
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data_augmentation import (
    get_train_transforms,
    get_val_transforms,
    Mixup,
    Cutmix,
)


# 标签名称映射
LABEL_NAMES = ["覆冰", "雪", "积雪", "霜冻"]
BINARY_LABEL_NAMES = ["冰雪异常"]
NUM_CLASSES = len(LABEL_NAMES)


def get_label_names(task: str = "multi_label") -> List[str]:
    """根据任务类型返回标签名。"""
    if task in {"binary", "binary_ice"}:
        return BINARY_LABEL_NAMES.copy()
    if task in {"multi_label", "multilabel", "multi"}:
        return LABEL_NAMES.copy()
    raise ValueError(f"不支持的任务类型: {task}")


def infer_label_names(data: List[Dict], fallback: Optional[List[str]] = None) -> List[str]:
    """从标注数据中推断标签名，兼容二分类和原四标签格式。"""
    if fallback:
        return list(fallback)

    for item in data:
        item_label_names = item.get("label_names")
        if item_label_names:
            return list(item_label_names)

        vector = item.get("label_vector")
        if vector is not None:
            if len(vector) == 1:
                return BINARY_LABEL_NAMES.copy()
            if len(vector) <= len(LABEL_NAMES):
                return LABEL_NAMES[:len(vector)]

    return LABEL_NAMES.copy()


class IceMultiLabelDataset(Dataset):
    """
    覆冰图像多标签数据集

    Args:
        label_file: 标注JSON文件路径
        transform: 数据变换
        image_dir: 图像目录（如果标注中使用相对路径）
    """

    def __init__(
        self,
        label_file: str,
        transform=None,
        image_dir: Optional[str] = None,
        strong_transform=None,
        standard_transform=None,
        label_names: Optional[List[str]] = None,
    ):
        self.transform = transform
        self.image_dir = image_dir
        self.strong_transform = strong_transform
        self.standard_transform = standard_transform
        self._filename_index = None

        # 加载标注
        with open(label_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.label_names = infer_label_names(self.data, label_names)
        self.num_classes = len(self.label_names)

        # 过滤无效数据
        valid_data = []
        missing_paths = []
        for item in self.data:
            image_path = self._resolve_image_path(item["image_path"])
            if os.path.exists(image_path):
                item = dict(item)
                item["image_path"] = image_path
                valid_data.append(item)
            elif len(missing_paths) < 5:
                missing_paths.append(item["image_path"])
        self.data = valid_data

        print(f"[Dataset] 加载 {len(self.data)} 张图像")
        if missing_paths:
            print("[Dataset] 警告: 部分标注图片路径不存在，已跳过。示例:")
            for path in missing_paths:
                print(f"  - {path}")

    def _resolve_image_path(self, image_path: str) -> str:
        candidates = []
        if os.path.isabs(image_path):
            candidates.append(image_path)
        else:
            candidates.append(image_path)

        if self.image_dir:
            candidates.append(os.path.join(self.image_dir, image_path))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        filename = PureWindowsPath(image_path).name or Path(image_path).name
        lookup = self._get_filename_index()
        if filename in lookup:
            return lookup[filename]

        return image_path

    def _get_filename_index(self) -> Dict[str, str]:
        """给迁移过的标注文件做兜底：按文件名回查本地图片。"""
        if self._filename_index is not None:
            return self._filename_index

        roots = []
        if self.image_dir:
            roots.append(Path(self.image_dir))
        roots.extend([Path("data/roboflow_train"), Path("data/imagine")])

        index = {}
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in suffixes:
                    index.setdefault(path.name, str(path))

        self._filename_index = index
        return index

    def __len__(self):
        return len(self.data)

    def _normalize_label_vector(self, label_vector: List[int]) -> List[int]:
        vector = [int(bool(v)) for v in label_vector]
        if len(vector) < self.num_classes:
            vector = vector + [0] * (self.num_classes - len(vector))
        return vector[:self.num_classes]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        image_path = item["image_path"]

        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 获取标签向量
        label_vector = item.get("label_vector")
        if label_vector is None:
            labels_dict = item.get("labels", {})
            label_vector = [int(bool(labels_dict.get(name, False))) for name in self.label_names]
        label_vector = self._normalize_label_vector(label_vector)
        labels = torch.tensor(label_vector, dtype=torch.float32)

        # 应用数据变换
        if self.strong_transform and self.standard_transform:
            transform = self.strong_transform if any(label_vector) else self.standard_transform
            image = transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, labels

    def get_label_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理类别不平衡）

        Returns:
            torch.Tensor: 每个类别的权重
        """
        # 统计每个类别的正样本数
        label_counts = torch.zeros(self.num_classes)
        for item in self.data:
            label_vector = item.get("label_vector", [0] * self.num_classes)
            label_counts += torch.tensor(self._normalize_label_vector(label_vector), dtype=torch.float32)

        # 计算pos_weight（正样本越少，权重越高），用于BCEWithLogitsLoss
        total = len(self.data)
        negative_counts = total - label_counts
        weights = negative_counts / (label_counts + 1e-6)
        return torch.clamp(weights, min=1.0, max=50.0)

    def get_sample_weights(self) -> torch.Tensor:
        """
        计算每个样本的权重（用于WeightedRandomSampler）

        Returns:
            torch.Tensor: 每个样本的权重
        """
        sample_weights = []
        for item in self.data:
            label_vector = self._normalize_label_vector(
                item.get("label_vector", [0] * self.num_classes)
            )
            # 如果有任何正标签，给予更高权重
            if any(label_vector):
                weight = 5.0  # 少数类样本权重更高
            else:
                weight = 1.0
            sample_weights.append(weight)

        return torch.tensor(sample_weights, dtype=torch.float32)


class IceImageFolderDataset(Dataset):
    """
    基于文件夹结构的覆冰图像数据集

    目录结构：
    data/dataset/
    ├── train/
    │   ├── 覆冰/
    │   ├── 雪/
    │   ├── 积雪/
    │   ├── 霜冻/
    │   └── 无/
    ├── val/
    └── test/

    Args:
        root_dir: 数据集根目录
        split: 数据集划分（train/val/test）
        transform: 数据变换
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        label_names: Optional[List[str]] = None,
    ):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.label_names = label_names or LABEL_NAMES
        self.num_classes = len(self.label_names)
        self.data = []

        # 标签映射
        if self.label_names == BINARY_LABEL_NAMES:
            label_map = {
                "冰雪异常": 0,
                "异常": 0,
                "有": 0,
                "positive": 0,
                "正常": -1,
                "无": -1,
                "negative": -1,
            }
        else:
            label_map = {
                "覆冰": 0,
                "雪": 1,
                "积雪": 2,
                "霜冻": 3,
                "无": -1,
            }

        # 扫描目录
        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            for filename in os.listdir(label_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(label_dir, filename)

                    # 创建多标签向量
                    label_vector = [0] * self.num_classes
                    if label_idx >= 0:
                        label_vector[label_idx] = 1

                    self.data.append({
                        "image_path": image_path,
                        "label_vector": label_vector,
                    })

        print(f"[Dataset] {split}: 加载 {len(self.data)} 张图像")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        image_path = item["image_path"]

        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 获取标签
        labels = torch.tensor(item["label_vector"], dtype=torch.float32)

        # 应用数据变换
        if self.transform:
            image = self.transform(image)

        return image, labels

    def get_sample_weights(self) -> torch.Tensor:
        """计算每个样本的权重（用于WeightedRandomSampler）。"""
        sample_weights = []
        for item in self.data:
            label_vector = item.get("label_vector", [0] * self.num_classes)
            sample_weights.append(5.0 if any(label_vector) else 1.0)
        return torch.tensor(sample_weights, dtype=torch.float32)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器

    Args:
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        batch_size: 批次大小
        num_workers: 工作线程数
        use_weighted_sampling: 是否使用加权采样

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 训练集加载器
    if use_weighted_sampling and hasattr(train_dataset, "get_sample_weights"):
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # 验证集加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 测试集加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_dataloaders(
    data_dir: str,
    label_file: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_strong_augment: bool = True,
    use_weighted_sampling: bool = True,
    label_names: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取数据加载器的便捷函数

    Args:
        data_dir: 数据目录
        label_file: 标注文件路径（如果使用基于标注的数据集）
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 工作线程数
        use_strong_augment: 是否对少数类使用强增强
        use_weighted_sampling: 是否使用加权采样

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 获取数据变换
    strong_train_transform = get_train_transforms(
        image_size=image_size,
        strong_augment=True,
    )
    standard_train_transform = get_train_transforms(
        image_size=image_size,
        strong_augment=False,
    )
    train_transform = strong_train_transform if use_strong_augment else standard_train_transform
    val_transform = get_val_transforms(image_size=image_size)

    # 创建数据集
    label_path = Path(label_file) if label_file else Path(data_dir)
    label_root = label_path.parent if label_path.is_file() else label_path
    use_label_json = label_root.is_dir() and all(
        (label_root / f"{split}_labels.json").exists()
        for split in ("train", "val", "test")
    )

    if use_label_json:
        # 基于标注文件的数据集
        train_dataset = IceMultiLabelDataset(
            label_file=str(label_root / "train_labels.json"),
            image_dir=data_dir,
            strong_transform=strong_train_transform if use_strong_augment else standard_train_transform,
            standard_transform=standard_train_transform,
            label_names=label_names,
        )
        val_dataset = IceMultiLabelDataset(
            label_file=str(label_root / "val_labels.json"),
            transform=val_transform,
            image_dir=data_dir,
            label_names=label_names,
        )
        test_dataset = IceMultiLabelDataset(
            label_file=str(label_root / "test_labels.json"),
            transform=val_transform,
            image_dir=data_dir,
            label_names=label_names,
        )
    else:
        if label_file:
            expected = ", ".join(f"{split}_labels.json" for split in ("train", "val", "test"))
            raise FileNotFoundError(
                f"标注目录缺少多标签划分文件: {label_root}，需要包含 {expected}"
            )
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"数据目录不存在: {data_dir}。请先运行 tools/split_dataset.py 生成 data/dataset。"
            )
        # 基于文件夹结构的数据集
        train_dataset = IceImageFolderDataset(
            root_dir=data_dir,
            split="train",
            transform=train_transform,
            label_names=label_names,
        )
        val_dataset = IceImageFolderDataset(
            root_dir=data_dir,
            split="val",
            transform=val_transform,
            label_names=label_names,
        )
        test_dataset = IceImageFolderDataset(
            root_dir=data_dir,
            split="test",
            transform=val_transform,
            label_names=label_names,
        )

    # 创建数据加载器
    return create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampling=use_weighted_sampling,
    )
