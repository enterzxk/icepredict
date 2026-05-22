"""
ice_monitor.alert — 融合预警决策模块

将时序预测结果与VLM图像识别结果进行交叉验证，输出分级预警。

预警等级定义：
  Level 0 — 正常：预测厚度<0.5mm 且 VLM=no
  Level 1 — 注意：预测厚度0.5~2mm 或 VLM识别结果与预测存在分歧
  Level 2 — 预警：预测厚度2~5mm 或 VLM=yes
  Level 3 — 紧急：预测厚度>5mm 且 VLM=yes 双重确认
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


LEVEL_NAMES = {
    0: "正常",
    1: "注意",
    2: "预警",
    3: "紧急",
}

LEVEL_COLORS = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red",
}

# 本地模型标签名称
LOCAL_LABEL_NAMES = ["覆冰", "雪", "积雪", "霜冻"]


@dataclass
class AlertResult:
    """预警结果"""
    level: int                        # 预警等级 0~3
    level_name: str                   # 等级名称
    color: str                        # 颜色标识

    ice_thickness: float              # 预测覆冰厚度 (mm)
    ice_ratio: float                  # 预测覆冰比值
    vlm_label: Optional[str] = None  # VLM识别结果 'yes'/'no'/'unknow'/None
    local_labels: Optional[List[str]] = None  # 本地模型识别结果 ['覆冰', '雪', ...]

    reason: str = ""                  # 预警原因说明

    def __str__(self):
        if self.local_labels:
            img_info = f"本地识别={'+'.join(self.local_labels)}"
        elif self.vlm_label:
            img_info = f"VLM={self.vlm_label}"
        else:
            img_info = "图像识别=未启用"
        return (
            f"[{self.level_name}] 覆冰厚度={self.ice_thickness:.2f}mm | "
            f"覆冰比值={self.ice_ratio:.3f} | {img_info} | {self.reason}"
        )

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "color": self.color,
            "ice_thickness_mm": round(self.ice_thickness, 4),
            "ice_ratio": round(self.ice_ratio, 4),
            "vlm_label": self.vlm_label,
            "local_labels": self.local_labels,
            "reason": self.reason,
        }


def fuse_alert(
    ice_thickness: float,
    ice_ratio: float,
    vlm_label: Optional[str] = None,
) -> AlertResult:
    """
    融合时序预测与VLM识别，生成预警结果

    Args:
        ice_thickness (float): 时序模型预测的覆冰厚度 (mm)
        ice_ratio (float):     时序模型预测的覆冰比值
        vlm_label (str|None):  VLM识别结果 ('yes'/'no'/'unknow'/None)
                               传入 None 表示未使用图像识别

    Returns:
        AlertResult
    """
    t = max(0.0, ice_thickness)  # 限制非负
    vlm = vlm_label

    # ——— 规则引擎 ———
    if vlm == "yes" and t > 5.0:
        level, reason = 3, f"时序预测={t:.1f}mm（>5mm）且VLM双重确认有覆冰"

    elif t > 5.0:
        level, reason = 3, f"时序预测={t:.1f}mm 严重超阈值（>5mm）"

    elif vlm == "yes" and t > 2.0:
        level, reason = 3, f"时序预测={t:.1f}mm（>2mm）且VLM图像确认有覆冰"

    elif vlm == "yes" or t > 2.0:
        level, reason = 2, (
            f"VLM图像识别发现覆冰，时序预测={t:.1f}mm" if vlm == "yes"
            else f"时序预测覆冰={t:.1f}mm 超过预警阈值（>2mm）"
        )

    elif t > 0.5 or (vlm == "yes" and t <= 0.5):
        level, reason = 1, (
            f"时序预测={t:.1f}mm（0.5~2mm间，需持续关注）"
            if not (vlm == "yes" and t <= 0.5)
            else f"VLM图像检测到轻微覆冰，但时序预测较低（{t:.1f}mm），存在分歧"
        )

    elif vlm == "unknow":
        level, reason = 1, f"VLM识别不确定，时序预测={t:.1f}mm，建议人工复核"

    else:
        level, reason = 0, f"时序预测={t:.1f}mm 正常，VLM未发现覆冰"

    return AlertResult(
        level=level,
        level_name=LEVEL_NAMES[level],
        color=LEVEL_COLORS[level],
        ice_thickness=t,
        ice_ratio=ice_ratio,
        vlm_label=vlm,
        reason=reason,
    )


def fuse_alert_local(
    ice_thickness: float,
    ice_ratio: float,
    local_labels: Optional[List[str]] = None,
) -> AlertResult:
    """
    融合时序预测与本地模型识别，生成预警结果

    Args:
        ice_thickness (float): 时序模型预测的覆冰厚度 (mm)
        ice_ratio (float):     时序模型预测的覆冰比值
        local_labels (list):   本地模型识别结果 ['覆冰', '雪', '积雪', '霜冻']
                               传入 None 或空列表表示未使用图像识别

    Returns:
        AlertResult
    """
    t = max(0.0, ice_thickness)  # 限制非负
    labels = local_labels or []

    # 判断是否有覆冰相关标签
    has_ice = "覆冰" in labels
    has_snow = "雪" in labels or "积雪" in labels
    has_frost = "霜冻" in labels
    has_any_ice = has_ice or has_snow or has_frost

    # ——— 规则引擎 ———
    if has_ice and t > 5.0:
        level, reason = 3, f"时序预测={t:.1f}mm（>5mm）且本地模型确认有覆冰"

    elif t > 5.0:
        level, reason = 3, f"时序预测={t:.1f}mm 严重超阈值（>5mm）"

    elif has_ice and t > 2.0:
        level, reason = 3, f"时序预测={t:.1f}mm（>2mm）且本地模型确认有覆冰"

    elif has_any_ice and t > 2.0:
        level, reason = 3, f"时序预测={t:.1f}mm（>2mm）且本地模型检测到{'、'.join(labels)}"

    elif has_ice or t > 2.0:
        level, reason = 2, (
            f"本地模型检测到覆冰，时序预测={t:.1f}mm" if has_ice
            else f"时序预测覆冰={t:.1f}mm 超过预警阈值（>2mm）"
        )

    elif has_any_ice and t <= 0.5:
        level, reason = 1, (
            f"本地模型检测到轻微{'、'.join(labels)}，但时序预测较低（{t:.1f}mm），存在分歧"
        )

    elif has_any_ice:
        level, reason = 2, f"本地模型检测到{'、'.join(labels)}，时序预测={t:.1f}mm"

    elif t > 0.5:
        level, reason = 1, f"时序预测={t:.1f}mm（0.5~2mm间，需持续关注）"

    else:
        level, reason = 0, f"时序预测={t:.1f}mm 正常，本地模型未发现覆冰"

    return AlertResult(
        level=level,
        level_name=LEVEL_NAMES[level],
        color=LEVEL_COLORS[level],
        ice_thickness=t,
        ice_ratio=ice_ratio,
        local_labels=labels,
        reason=reason,
    )


def fuse_alert_unified(
    ice_thickness: float,
    ice_ratio: float,
    vlm_label: Optional[str] = None,
    local_labels: Optional[List[str]] = None,
    use_local: bool = True,
) -> AlertResult:
    """
    统一的融合预警函数，支持VLM和本地模型

    Args:
        ice_thickness (float): 时序模型预测的覆冰厚度 (mm)
        ice_ratio (float):     时序模型预测的覆冰比值
        vlm_label (str|None):  VLM识别结果
        local_labels (list):   本地模型识别结果
        use_local (bool):      是否优先使用本地模型

    Returns:
        AlertResult
    """
    if use_local and local_labels is not None:
        return fuse_alert_local(ice_thickness, ice_ratio, local_labels)
    elif vlm_label:
        return fuse_alert(ice_thickness, ice_ratio, vlm_label)
    else:
        return fuse_alert(ice_thickness, ice_ratio, None)
