"""
ice_monitor.predictor
覆冰厚度时序预测模块 — 封装 Seq2ABTransformer 的加载与推理
"""

import os
import sys
import numpy as np
import torch

# 保证无论从哪里调用都能找到 src/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model import Seq2ABTransformer

# 默认权重路径
_DEFAULT_WEIGHTS = os.path.join(_PROJECT_ROOT, 'weights', '20251205', 'best.ckpt')
_CONDITION_LENGTH = 1440   # 2天 × 24小时 × 10条/小时 × 3特征
_MAX_PRED_WINDOW = 480     # 最多向前预测480步（48小时）


class IceThicknessPredictor:
    """
    覆冰厚度预测器

    使用方法：
        predictor = IceThicknessPredictor()
        result = predictor.predict(history_seq, future_conditions)

    参数：
        weights_path (str): 模型权重路径，默认使用 weights/20251205/best.ckpt
        device (str): 推理设备，默认自动检测 cuda/cpu
    """

    def __init__(self, weights_path: str = None, device: str = None):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        weights_path = weights_path or _DEFAULT_WEIGHTS

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"模型权重文件不存在: {weights_path}\n"
                f"请确认权重文件已放置在 weights/20251205/best.ckpt"
            )

        self.model = Seq2ABTransformer(condition_length=_CONDITION_LENGTH).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict)
        self.model.eval()

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[Predictor] 模型加载成功 | 设备: {self.device} | 参数量: {total:,}")

    @torch.no_grad()
    def predict(self, history_seq: np.ndarray, future_conditions: np.ndarray) -> dict:
        """
        预测覆冰厚度和覆冰比值

        Args:
            history_seq (np.ndarray): 历史时序，shape=(seq_len, 5)
                列顺序：[覆冰厚度, 覆冰比值, 温度, 湿度, 时间戳归一化]
            future_conditions (np.ndarray): 未来气象条件，shape=(pred_steps, 3)
                列顺序：[温度, 湿度, 时间戳归一化]
                pred_steps 不能超过 480（即48小时）

        Returns:
            dict: {
                'ice_thickness': float,   # 预测覆冰厚度 (mm)
                'ice_ratio':    float,    # 预测覆冰比值
                'pred_steps':   int,      # 预测的步数（时间窗口）
            }
        """
        if history_seq.ndim != 2 or history_seq.shape[1] != 5:
            raise ValueError(f"history_seq 应为 (seq_len, 5)，得到 {history_seq.shape}")
        if future_conditions.ndim != 2 or future_conditions.shape[1] != 3:
            raise ValueError(f"future_conditions 应为 (pred_steps, 3)，得到 {future_conditions.shape}")

        pred_steps = future_conditions.shape[0]
        if pred_steps > _MAX_PRED_WINDOW:
            raise ValueError(f"pred_steps={pred_steps} 超过最大预测窗口 {_MAX_PRED_WINDOW}")

        # 构建模型输入
        pre = torch.from_numpy(history_seq.astype(np.float32)).to(self.device).unsqueeze(0)

        cdt_flat = future_conditions.reshape(1, -1).astype(np.float32)
        pad_len = _CONDITION_LENGTH - cdt_flat.shape[1]
        if pad_len < 0:
            raise ValueError("future_conditions 展平后超过条件长度限制")
        cdt_flat = np.concatenate([cdt_flat, np.zeros((1, pad_len), dtype=np.float32)], axis=1)
        cdt = torch.from_numpy(cdt_flat).to(self.device)

        logit = self.model(pre, cdt)  # (1, 2)

        return {
            'ice_thickness': float(logit[0, 0].item()),
            'ice_ratio':     float(logit[0, 1].item()),
            'pred_steps':    pred_steps,
        }

    def predict_batch(self, samples: list) -> list:
        """
        批量预测

        Args:
            samples: list of (history_seq, future_conditions) tuples

        Returns:
            list of dicts，每个 dict 同 predict() 的返回格式
        """
        return [self.predict(h, c) for h, c in samples]

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
