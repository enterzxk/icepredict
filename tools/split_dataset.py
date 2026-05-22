"""
数据集划分工具
将标注数据划分为训练集、验证集、测试集
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

LABEL_NAMES = ["覆冰", "雪", "积雪", "霜冻"]
BINARY_LABEL_NAMES = ["冰雪异常"]


def load_labels(label_file: str) -> List[Dict]:
    """加载标注文件"""
    with open(label_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_label_names(data: List[Dict]) -> List[str]:
    """从标注文件中推断标签名。"""
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


def _label_vector(item: Dict, label_names: List[str] = None) -> List[int]:
    """从标注项中获取稳定的标签向量，兼容二分类和四标签。"""
    label_names = label_names or _infer_label_names([item])
    if "label_vector" in item:
        vector = item["label_vector"]
        vector = [int(bool(v)) for v in vector]
        if len(vector) < len(label_names):
            vector = vector + [0] * (len(label_names) - len(vector))
        return vector[:len(label_names)]

    labels = item.get("labels", {})
    return [int(bool(labels.get(name, False))) for name in label_names]


def _ensure_label_vector(item: Dict, label_names: List[str] = None) -> Dict:
    label_names = label_names or _infer_label_names([item])
    normalized = dict(item)
    normalized["label_vector"] = _label_vector(item, label_names)
    normalized["label_names"] = list(label_names)
    if "labels" not in normalized:
        normalized["labels"] = {
            name: bool(normalized["label_vector"][idx])
            for idx, name in enumerate(label_names)
        }
    return normalized


def _allocate_counts(total: int, ratios: Tuple[float, float, float]) -> List[int]:
    """按比例分配计数；样本足够时尽量保证每个子集至少1个。"""
    raw = [total * ratio for ratio in ratios]
    counts = [int(x) for x in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - counts[i], reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1

    if total >= len(ratios):
        for idx, count in enumerate(counts):
            if count == 0:
                donor = max(range(len(counts)), key=lambda i: counts[i])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    return counts


def stratified_split(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分层划分数据集，确保各类别在各子集中的比例一致

    Args:
        data: 标注数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_data, val_data, test_data)
    """
    random.seed(seed)

    ratios = (train_ratio, val_ratio, test_ratio)
    split_names = ("train", "val", "test")
    target_sizes = dict(zip(split_names, _allocate_counts(len(data), ratios)))

    label_names = _infer_label_names(data)
    num_classes = len(label_names)
    normalized_data = [_ensure_label_vector(item, label_names) for item in data]
    random.shuffle(normalized_data)
    normalized_data.sort(key=lambda item: sum(_label_vector(item, label_names)), reverse=True)

    label_totals = [0] * num_classes
    for item in normalized_data:
        for idx, value in enumerate(_label_vector(item, label_names)):
            label_totals[idx] += value

    target_label_counts = {
        split: [
            _allocate_counts(total, ratios)[split_idx]
            for total in label_totals
        ]
        for split_idx, split in enumerate(split_names)
    }

    splits = {name: [] for name in split_names}
    current_label_counts = {name: [0] * num_classes for name in split_names}

    for item in normalized_data:
        vector = _label_vector(item, label_names)

        def split_score(split_name: str) -> float:
            remaining_slots = target_sizes[split_name] - len(splits[split_name])
            if remaining_slots <= 0:
                return float("-inf")

            positive_need = sum(
                max(0, target_label_counts[split_name][idx] - current_label_counts[split_name][idx])
                for idx, value in enumerate(vector)
                if value
            )
            size_need = remaining_slots / max(1, target_sizes[split_name])
            no_label_bonus = size_need if not any(vector) else 0
            return positive_need * 10 + size_need + no_label_bonus

        chosen_split = max(split_names, key=split_score)
        splits[chosen_split].append(item)
        for idx, value in enumerate(vector):
            current_label_counts[chosen_split][idx] += value

    # 打乱顺序
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def create_symlinks(
    data: List[Dict],
    output_dir: str,
    split_name: str,
):
    """
    创建符号链接或复制图像到目标目录

    Args:
        data: 数据列表
        output_dir: 输出根目录
        split_name: 子集名称（train/val/test）
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    label_names = _infer_label_names(data)

    # 按标签创建子目录
    label_dirs = {}
    for label in [*label_names, "无"]:
        label_dir = os.path.join(split_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        label_dirs[label] = label_dir

    for item in data:
        src_path = item["image_path"]
        vector = _label_vector(item, label_names)

        # 确定主标签
        primary_label = "无"
        for idx, val in enumerate(vector):
            if val:
                primary_label = label_names[idx]
                break

        # 创建目标路径
        filename = os.path.basename(src_path)
        dst_path = os.path.join(label_dirs[primary_label], filename)

        # 创建符号链接（Windows需要管理员权限或开发者模式）
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(os.path.abspath(src_path), dst_path)
        except OSError:
            # 如果符号链接失败，复制文件
            import shutil
            shutil.copy2(src_path, dst_path)


def save_split_info(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str,
):
    """保存划分信息"""
    split_info = {
        "train": {
            "count": len(train_data),
            "files": [item["image_path"] for item in train_data],
        },
        "val": {
            "count": len(val_data),
            "files": [item["image_path"] for item in val_data],
        },
        "test": {
            "count": len(test_data),
            "files": [item["image_path"] for item in test_data],
        },
    }
    label_names = _infer_label_names(train_data + val_data + test_data)

    # 统计每个子集的标签分布
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        stats = {label: 0 for label in label_names}
        stats["无"] = 0
        for item in split_data:
            labels = _ensure_label_vector(item, label_names).get("labels", {})
            has_any = False
            for key in label_names:
                val = labels.get(key, False)
                if val:
                    stats[key] += 1
                    has_any = True
            if not has_any:
                stats["无"] += 1
        split_info[split_name]["label_stats"] = stats

    info_file = os.path.join(output_dir, "split_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        label_file = os.path.join(output_dir, f"{split_name}_labels.json")
        normalized = [_ensure_label_vector(item, label_names) for item in split_data]
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"\n划分信息已保存至: {info_file}")

    # 打印统计
    print("\n=== 数据集划分统计 ===")
    print(f"训练集: {len(train_data)} 张")
    print(f"验证集: {len(val_data)} 张")
    print(f"测试集: {len(test_data)} 张")
    print(f"总计: {len(train_data) + len(val_data) + len(test_data)} 张")

    print("\n=== 各子集标签分布 ===")
    for split_name in ["train", "val", "test"]:
        stats = split_info[split_name]["label_stats"]
        print(f"\n{split_name}:")
        for label, count in stats.items():
            total = split_info[split_name]["count"]
            print(f"  {label}: {count} ({count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument(
        "--label-file",
        type=str,
        default="data/labels/training_labels.json",
        help="标注文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset",
        help="输出目录"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="测试集比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="不创建符号链接，只保存划分信息"
    )
    args = parser.parse_args()

    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"错误: 比例之和必须为1.0，当前为 {total_ratio}")
        return

    # 加载数据
    if not os.path.exists(args.label_file):
        print(f"错误: 标注文件不存在: {args.label_file}")
        return

    data = load_labels(args.label_file)
    print(f"加载标注数据: {len(data)} 条")

    # 划分数据集
    train_data, val_data, test_data = stratified_split(
        data,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建符号链接
    if not args.no_symlinks:
        print("\n创建数据集目录...")
        create_symlinks(train_data, args.output_dir, "train")
        create_symlinks(val_data, args.output_dir, "val")
        create_symlinks(test_data, args.output_dir, "test")
        print("数据集目录创建完成")

    # 保存划分信息
    save_split_info(train_data, val_data, test_data, args.output_dir)


if __name__ == "__main__":
    main()
