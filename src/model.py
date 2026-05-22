import torch
import torch.nn as nn
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from einops import rearrange
from .model_utils import DropPath, SinusoidalPositionalEncoding

try:
    from xformers.ops import memory_efficient_attention as _xformers_attention
except ImportError:
    _xformers_attention = None


def _memory_efficient_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if _xformers_attention is not None:
        try:
            return _xformers_attention(
                q.bfloat16().contiguous(),
                k.bfloat16().contiguous(),
                v.bfloat16().contiguous(),
            ).float()
        except (NotImplementedError, RuntimeError):
            pass

    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    message = F.scaled_dot_product_attention(q, k, v)
    return message.permute(0, 2, 1, 3).contiguous()

# 1. 时间戳编码（处理tn、tn-1等时间特征）
class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        # 周期编码：捕捉时间的日/周/年周期（可根据数据粒度调整）
        self.time_dim = time_dim
        self.freqs = nn.Parameter(torch.randn(time_dim // 2))  # 可学习频率
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        输入：t - 原始时间戳，shape=(batch_size, 1)
        输出：time_feat - 编码后的时间特征，shape=(batch_size, time_dim)
        """
        t = t / (3600 * 24)  # 归一化到“天”（根据数据粒度调整）
        # 正弦+余弦周期编码（类似Transformer位置编码）
        rads = t * self.freqs.unsqueeze(0)
        sin_feat = torch.sin(rads)
        cos_feat = torch.cos(rads)
        time_feat = torch.cat([sin_feat, cos_feat], dim=-1)
        return time_feat

# 2. 基础MLP模块（复用）
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hid_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hid_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hid_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,mlp_ratio=4,**kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.pre_norm = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = FullAttention()
        
        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*mlp_ratio,d_model),
            nn.Dropout(dropout)
        )
        self.dropout = DropPath(dropout)
        
    def forward(self, x0,x1=None):
        '''
        x0,x1: [n, l, d]
        mask0,mask1: [n, l]
        '''
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_norm(x0), self.pre_norm(x1)
        q = self.q_proj(_x0)
        k = self.k_proj(_x1)
        v = self.v_proj(_x1)
        
        q,k,v = map(lambda x : rearrange(x,'n l (h d) -> n l h d',h=self.nhead),[q,k,v])
        message = _memory_efficient_attention(q, k, v)
        message = rearrange(message,'n l h d -> n l (h d)')
        
        x = x0 + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x

class Seq2ABTransformer(nn.Module):
    def __init__(
        self,
        x_dim: int = 5,  # 单步x的维度：a/b/c/d/t
        time_dim: int = 16,  # 时间编码维度
        feat_dim: int = 64,  # MLP1输出特征维度（f_n-k）
        n_head: int = 8,  # 注意力头数
        n_layers: int = 3,  # Transformer Encoder层数
        ff_dim: int = 128,  # Transformer前馈网络维度
        dropout: float = 0.1,
        condition_length = 3 * 24 * 6 * 3 # day * hour * record_per_hour * cdt_per_record
    ):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 1. MLP1：映射单步x到f（历史序列特征提升）
        self.mlp1 = MLP(in_dim=x_dim, hidden_dims=[32], out_dim=feat_dim, dropout=dropout)
        
        # 2. 可学习的CLS Token（聚合序列全局信息）
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))  # shape=(1,1,feat_dim)
        self.pos = SinusoidalPositionalEncoding(d_model=feat_dim, max_len=41)

        # 3. Transformer Encoder多层堆叠
        self.transformer_layers = nn.ModuleList([
            AttentionLayer(d_model=feat_dim,d_head=feat_dim//n_head,dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 4. 时间编码模块（处理tn）
        # self.time_encoder = TimeEncoder(time_dim=time_dim)
        
        # 5. MLP2：编码当前条件特征（cn/dn/tn）
        # 输入维度：cn(1)+dn(1)+time_feat(time_dim) → 总维度=2+time_dim
        self.mlp2 = MLP(condition_length, hidden_dims=[32], out_dim=feat_dim, dropout=dropout)
        
        # 6. MLP3：融合CLS特征+条件特征，预测an/bn
        self.mlp3 = MLP(in_dim=2 * feat_dim, hidden_dims=[64, 32], out_dim=2, dropout=dropout)
        
        # 7. 层归一化（提升训练稳定性）
        self.norm = nn.LayerNorm(feat_dim)
        

    def forward(
        self,
        hist_x: torch.Tensor,  # 历史序列x_{n-1}~x_{n-m}，shape=(batch_size, m, x_dim)
        curr_cdt: torch.Tensor  # 当前条件特征[cn, dn, tn]，shape=(batch_size, 3)
    ) -> torch.Tensor:
        """
        返回：预测的an/bn，shape=(batch_size, 2)
        """
        batch_size = hist_x.shape[0]
        
        # -------------------------- 第一步：处理历史序列 --------------------------
        # 1. MLP1映射：每个历史步x→f
        f_seq = self.mlp1(hist_x)  # shape=(batch_size, m, feat_dim)
        
        # 2. 添加CLS Token到序列头部
        # 扩展CLS Token到batch维度：(1,1,feat_dim) → (batch_size,1,feat_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 拼接：[CLS, f_{n-1}, f_{n-2}, ..., f_{n-m}] → shape=(batch_size, m+1, feat_dim)
        transformer_input = torch.cat([cls_tokens, f_seq], dim=1)
        
        # 3. 多层Transformer Encoder处理
        x = self.norm(transformer_input)
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 4. 提取CLS Token的输出（聚合后的序列特征）
        cls_feat = x[:, 0, :]  # shape=(batch_size, feat_dim)
        
        # -------------------------- 第二步：处理当前条件特征 --------------------------
        cdt_feat = self.mlp2(curr_cdt)  # shape=(batch_size, feat_dim)
        
        # -------------------------- 第三步：融合特征并预测 --------------------------
        # 拼接CLS特征和条件特征
        fusion_feat = torch.cat([cls_feat, cdt_feat], dim=-1)  # shape=(batch_size, 2*feat_dim)
        
        # MLP3预测an/bn
        ab_pred = self.mlp3(fusion_feat)  # shape=(batch_size, 2)
        
        return ab_pred
