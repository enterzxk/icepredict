"""
数据增强模块
针对覆冰图像识别任务的多标签分类数据增强
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import Tuple, List, Optional


class Cutout:
    """随机遮挡增强"""

    def __init__(self, n_holes: int = 1, length: int = 40):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img


class RandomErasing:
    """随机擦除增强（比Cutout更灵活）"""

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.2,
        r1: float = 0.3,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                if img.size(0) == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[:, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_train_transforms(
    image_size: int = 224,
    strong_augment: bool = True,
) -> T.Compose:
    """
    获取训练集数据增强pipeline

    Args:
        image_size: 目标图像尺寸
        strong_augment: 是否使用强增强（用于少数类）

    Returns:
        torchvision.transforms.Compose
    """
    if strong_augment:
        # 强增强策略（用于覆冰/雪/积雪/霜冻类）
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomRotation(30),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.8, 1.0),
                shear=10,
            ),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            Cutout(n_holes=1, length=40),
        ])
    else:
        # 标准增强策略（用于无覆冰类）
        return T.Compose([
            T.Resize((image_size + 16, image_size + 16)),
            T.RandomCrop(image_size),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """
    获取验证/测试集数据变换

    Args:
        image_size: 目标图像尺寸

    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inference_transforms(image_size: int = 224) -> T.Compose:
    """
    获取推理时的数据变换（与验证集相同）

    Args:
        image_size: 目标图像尺寸

    Returns:
        torchvision.transforms.Compose
    """
    return get_val_transforms(image_size)


class Mixup:
    """Mixup数据增强"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对batch进行Mixup

        Args:
            images: 图像tensor (B, C, H, W)
            labels: 标签tensor (B, num_classes)

        Returns:
            mixed_images, mixed_labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels


class Cutmix:
    """Cutmix数据增强"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对batch进行Cutmix

        Args:
            images: 图像tensor (B, C, H, W)
            labels: 标签tensor (B, num_classes)

        Returns:
            mixed_images, mixed_labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        # 生成随机bbox
        _, _, h, w = images.size()
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cx = np.random.randint(h)
        cy = np.random.randint(w)

        x1 = np.clip(cx - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_h // 2, 0, h)
        y1 = np.clip(cy - cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_w // 2, 0, w)

        # 应用Cutmix
        mixed_images = images.clone()
        mixed_images[:, :, x1:x2, y1:y2] = images[index, :, x1:x2, y1:y2]

        # 调整lambda
        lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels


def visualize_augmentation(
    image_path: str,
    num_samples: int = 8,
    save_path: str = "augmentation_samples.png",
):
    """
    可视化数据增强效果

    Args:
        image_path: 原始图像路径
        num_samples: 生成样本数量
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # 加载原始图像
    image = Image.open(image_path).convert("RGB")

    # 获取增强变换
    strong_transform = get_train_transforms(strong_augment=True)
    standard_transform = get_train_transforms(strong_augment=False)

    # 生成增强样本
    fig, axes = plt.subplots(2, num_samples // 2 + 1, figsize=(15, 6))

    # 显示原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    # 显示强增强样本
    for i in range(num_samples // 2):
        augmented = strong_transform(image)
        # 反归一化用于显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_show = augmented * std + mean
        img_show = img_show.permute(1, 2, 0).numpy().clip(0, 1)

        axes[0, i + 1].imshow(img_show)
        axes[0, i + 1].set_title(f"Strong {i+1}")
        axes[0, i + 1].axis("off")

    # 显示标准增强样本
    for i in range(num_samples // 2):
        augmented = standard_transform(image)
        img_show = augmented * std + mean
        img_show = img_show.permute(1, 2, 0).numpy().clip(0, 1)

        axes[1, i + 1].imshow(img_show)
        axes[1, i + 1].set_title(f"Standard {i+1}")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"增强样本已保存至: {save_path}")
