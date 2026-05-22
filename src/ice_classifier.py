"""
覆冰图像分类器模型
基于ResNet50的多标签分类模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Dict


class IceMultiLabelClassifier(nn.Module):
    """
    覆冰图像多标签分类器

    基于ResNet50 backbone + 自定义分类头
    输出4个独立的sigmoid概率，分别对应：覆冰、雪、积雪、霜冻

    Args:
        num_classes: 类别数量（默认4）
        pretrained: 是否使用预训练权重
        dropout: Dropout比率
        freeze_backbone: 是否冻结backbone
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        # 加载预训练ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)

        # 获取backbone输出维度
        backbone_out_features = self.backbone.fc.in_features

        # 移除原始分类头
        self.backbone.fc = nn.Identity()

        # 冻结backbone（如果需要）
        if freeze_backbone:
            self._freeze_backbone()

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        # 初始化分类头权重
        self._init_classifier_weights()

    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone已冻结")

    def _unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone已解冻")

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像tensor (B, C, H, W)

        Returns:
            torch.Tensor: 多标签logits (B, num_classes)
        """
        # 提取特征
        features = self.backbone(x)

        # 分类
        logits = self.classifier(features)

        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict:
        """
        预测单张图像

        Args:
            x: 输入图像tensor (1, C, H, W)
            threshold: 分类阈值

        Returns:
            dict: {
                'logits': torch.Tensor,
                'probabilities': torch.Tensor,
                'predictions': torch.Tensor,
                'labels': list,
            }
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).int()

            # 获取预测标签
            labels = []
            label_names = ["冰雪异常"] if self.num_classes == 1 else ["覆冰", "雪", "积雪", "霜冻"]
            for i in range(self.num_classes):
                if predictions[0, i] == 1:
                    labels.append(label_names[i])

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "labels": labels,
        }

    def freeze_backbone(self):
        """冻结backbone（用于第一阶段训练）"""
        self._freeze_backbone()

    def unfreeze_backbone(self):
        """解冻backbone（用于第二阶段训练）"""
        self._unfreeze_backbone()

    def get_param_groups(self, lr: float = 1e-3, weight_decay: float = 1e-4) -> List[Dict]:
        """
        获取参数组（用于优化器）

        Args:
            lr: 学习率
            weight_decay: 权重衰减

        Returns:
            list of dict: 参数组
        """
        # backbone参数（较低学习率）
        backbone_params = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # 分类头参数（较高学习率）
        classifier_params = list(self.classifier.parameters())

        param_groups = [
            {
                "params": backbone_params,
                "lr": lr * 0.1,  # backbone学习率较低
                "name": "backbone",
            },
            {
                "params": classifier_params,
                "lr": lr,
                "name": "classifier",
            },
        ]

        return param_groups


class IceMultiLabelClassifierV2(nn.Module):
    """
    覆冰图像多标签分类器V2

    使用更复杂的注意力机制和特征融合

    Args:
        num_classes: 类别数量（默认4）
        pretrained: 是否使用预训练权重
        dropout: Dropout比率
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes

        # 加载预训练ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)

        # 获取backbone输出维度
        backbone_out_features = self.backbone.fc.in_features

        # 移除原始分类头
        self.backbone.fc = nn.Identity()

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_features, backbone_out_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_features // 16, backbone_out_features),
            nn.Sigmoid(),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.3),
            nn.Linear(512, num_classes),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像tensor (B, C, H, W)

        Returns:
            torch.Tensor: 多标签logits (B, num_classes)
        """
        # 提取特征
        features = self.backbone(x)

        # 应用通道注意力
        attention_weights = self.channel_attention(features.unsqueeze(-1))
        features = features * attention_weights

        # 分类
        logits = self.classifier(features)

        return logits


def create_model(
    model_name: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    创建模型的便捷函数

    Args:
        model_name: 模型名称（resnet50, resnet50v2）
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        dropout: Dropout比率
        freeze_backbone: 是否冻结backbone

    Returns:
        nn.Module: 模型
    """
    if model_name == "resnet50":
        model = IceMultiLabelClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    elif model_name == "resnet50v2":
        model = IceMultiLabelClassifierV2(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    return model


def load_model(
    checkpoint_path: str,
    model_name: str = "resnet50",
    num_classes: int = 4,
    device: str = "cpu",
) -> nn.Module:
    """
    加载模型checkpoint

    Args:
        checkpoint_path: checkpoint文件路径
        model_name: 模型名称
        num_classes: 类别数量
        device: 设备

    Returns:
        nn.Module: 加载了权重的模型
    """
    # 创建模型
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理不同的checkpoint格式
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"[Model] 从 {checkpoint_path} 加载成功")

    return model
