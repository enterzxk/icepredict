import torch
import torch.nn as nn
import math
from einops import rearrange
from kornia.utils import create_meshgrid

class PosEmb(nn.Module):
    def __init__(self,d_model=128,temperature=0.1,k=100,norm=False):
        super().__init__()
        self.temperature=temperature
        self.k = k
        self.embedding_linear = torch.nn.Linear(k*2, d_model)
        self.norm = norm

    def forward(self,x):
        x0,x1 = torch.chunk(x, chunks=2, dim=0)
        n,c,h,w = x0.shape
        f0 = rearrange(x0, 'n c h w -> n (h w) c')
        f1 = rearrange(x1, 'n c h w -> n (h w) c')
        f0_norm = f0 / f0.shape[-1] ** 0.5
        f1_norm = f1 / f1.shape[-1] ** 0.5
        sm = torch.einsum('nld,nsd->nls',f0_norm,f1_norm) / self.temperature
        cm = (torch.softmax(sm, dim=1) * torch.softmax(sm, dim=2)).view(n,-1)
        _,topi = torch.topk(cm,k=self.k,dim=-1)
        qindex = torch.div(topi,h*w,rounding_mode='trunc')
        rindex = topi % (h*w)
        coords0 = torch.stack([qindex % w, torch.div(qindex, w,rounding_mode='trunc')],dim=-1) / w
        coords1 = torch.stack([rindex % w, torch.div(rindex, w,rounding_mode='trunc')],dim=-1) / w
        grid = create_meshgrid(h, w, False,device=x0.device).reshape(1,-1,2) / w
        offsets0 = grid[:,:,None,:] - coords0[:,None,:,:]
        offsets1 = grid[:,:,None,:] - coords1[:,None,:,:]
        if self.norm:
            scale0 = torch.sqrt(torch.pow(coords0[:,:,None,:] - coords0[:,None,:,:],2).sum(dim=-1,keepdims=True)).mean(dim=(1,2,3),keepdims=True)
            scale1 = torch.sqrt(torch.pow(coords1[:,:,None,:] - coords1[:,None,:,:],2).sum(dim=-1,keepdims=True)).mean(dim=(1,2,3),keepdims=True)
            offsets0 = offsets0 / (scale0 / scale1)
        embedding0 = self.embedding_linear(offsets0.flatten(2)) 
        embedding1 = self.embedding_linear(offsets1.flatten(2))
        f0 = f0 + embedding0
        f1 = f1 + embedding1
        f = rearrange(torch.cat([f0,f1],dim=0),'n (h w) d -> n d h w',h=h)
        return f


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        """
        Args:
            d_model: 特征维度（与feat_dim一致）
            max_len: 最大序列长度（需≥m+1，m为历史步数，+1是CLS Token）
        """
        super().__init__()
        # 初始化位置编码矩阵：(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置序列：(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 频率因子：10000^(2i/d_model)，i为特征维度索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0,2,4...维度
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # 1,3,5...维度（处理奇数d_model）
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        输入：seq_len - 当前序列长度（m+1，包含CLS Token）
        输出：pos_encoding - 位置编码，shape=(1, seq_len, d_model)（适配batch维度）
        """
        return self.pe[:seq_len].unsqueeze(0)  # 扩展batch维度，方便广播
    
class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256),norm=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if norm:
            y_position = y_position / max_shape[0]
            x_position = x_position / max_shape[1]
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class PositionEncodingSineNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape
        self.pe = None

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        if self.pe is None or self.pe.shape[2] != x.shape[2] or self.pe.shape[3] != x.shape[3]:
            pe = torch.zeros((self.d_model, x.shape[2], x.shape[3]))
            y_position = torch.ones((x.shape[2], x.shape[3])).cumsum(0).float().unsqueeze(0) * self.max_shape[0] / x.shape[2]
            x_position = torch.ones((x.shape[2], x.shape[3])).cumsum(1).float().unsqueeze(0) * self.max_shape[1] / x.shape[3]

            div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2)))
            div_term = div_term[:, None, None]  # [C//4, 1, 1]
            pe[0::4, :, :] = torch.sin(x_position * div_term)
            pe[1::4, :, :] = torch.cos(x_position * div_term)
            pe[2::4, :, :] = torch.sin(y_position * div_term)
            pe[3::4, :, :] = torch.cos(y_position * div_term)
            self.pe = pe.unsqueeze(0).to(x.device)

        return x + self.pe

class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, dim, max_shape=(152,152)):
        super().__init__()
        self.dim = dim
        self.max_shape = max_shape
        
        
        # 计算所有可能的角度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / (dim//4)))
        h,w = max_shape
        x_seq = torch.arange(w).float()
        y_seq = torch.arange(h).float()
        x_inp = torch.einsum("i,j->ij", x_seq, inv_freq)
        y_inp = torch.einsum("i,j->ij", y_seq, inv_freq)
        
        x_inp = torch.cat([x_inp,x_inp],dim=-1)
        y_inp = torch.cat([y_inp,y_inp],dim=-1)
        x_sin_emb = x_inp.sin()
        x_cos_emb = x_inp.cos()
        y_sin_emb = y_inp.sin()
        y_cos_emb = y_inp.cos()
        
        self.register_buffer('cos_emb', x_cos_emb.T.unsqueeze(-2) + y_cos_emb.T.unsqueeze(-1))
        self.register_buffer('sin_emb', x_sin_emb.T.unsqueeze(-2) + y_sin_emb.T.unsqueeze(-1))

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=2)
        return torch.cat((-x2, x1), dim=2)

    def forward(self, x):
        n,heads,c,h,w = x.shape
        cos_emb = self.cos_emb[...,:h,:w]
        sin_emb = self.sin_emb[...,:h,:w]
        # 应用旋转操作
        x_rotated = x * cos_emb + self.rotate_half(x) * sin_emb
        
        return x_rotated
    
class RotaryPositionEmbedding2DAdd(nn.Module):
    def __init__(self, dim, max_shape=(200,200),method='add_theta'):
        super().__init__()
        self.dim = dim
        self.max_shape = max_shape
        
        inv_freq_x = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq_y = 1.0 / (1000 ** (torch.arange(0, dim, 2).float() / dim))
        h,w = max_shape
        x_seq = torch.arange(w).float()
        y_seq = torch.arange(h).float()
        x_inp = torch.einsum("i,j->ij", inv_freq_x, x_seq)
        y_inp = torch.einsum("i,j->ij", inv_freq_y, y_seq)
        
        x_inp = torch.cat([x_inp,x_inp],dim=0).unsqueeze(1) # (d, 1, w)
        y_inp = torch.cat([y_inp,y_inp],dim=0).unsqueeze(2) # (d, h, 1)
        if method == 'add_theta':
            inp = x_inp + y_inp
            self.register_buffer('cos_emb', inp.cos())
            self.register_buffer('sin_emb', inp.sin())
        else:
            self.register_buffer("cos_emb", x_inp.cos() + y_inp.cos())
            self.register_buffer("sin_emb", x_inp.sin() + y_inp.sin())

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=2)
        return torch.cat((-x2, x1), dim=2)

    def forward(self, x):
        n,heads,c,h,w = x.shape
        cos_emb = self.cos_emb[...,:h,:w]
        sin_emb = self.sin_emb[...,:h,:w]
        # 应用旋转操作
        x_rotated = x * cos_emb + self.rotate_half(x) * sin_emb
        
        return x_rotated

class RotaryPositionEmbedding2DCat(nn.Module):
    def __init__(self, dim, max_shape=(152,152)):
        super().__init__()
        self.dim = dim
        self.max_shape = max_shape
        h,w = max_shape
        x_seq = torch.arange(w).float()
        y_seq = torch.arange(h).float()
        inv_freq_x = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        inv_freq_y = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        x_inp = torch.einsum("i,j->ij", inv_freq_x, x_seq)
        y_inp = torch.einsum("i,j->ij", inv_freq_y, y_seq)

        x_inp = torch.cat([x_inp,x_inp],dim=0) # (d//2, w)
        y_inp = torch.cat([y_inp,y_inp],dim=0) # (d//2, h)
        x_emb_cos = x_inp.cos().unsqueeze(1)
        x_emb_sin = x_inp.sin().unsqueeze(1)
        y_emb_cos = y_inp.cos().unsqueeze(-1)
        y_emb_sin = y_inp.sin().unsqueeze(-1)
        self.register_buffer("cos_emb",torch.cat([torch.tile(x_emb_cos,(1,h,1)),torch.tile(y_emb_cos,(1,1,w))],dim=0))
        self.register_buffer("sin_emb",torch.cat([torch.tile(x_emb_sin,(1,h,1)),torch.tile(y_emb_sin,(1,1,w))],dim=0))

    def rotate_quater(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=2)
        return torch.cat((-x2, x1, -x4, x3), dim=2)

    def forward(self, x):
        n,heads,c,h,w = x.shape
        cos_emb = self.cos_emb[...,:h,:w]
        sin_emb = self.sin_emb[...,:h,:w]
        # 应用旋转操作
        x_rotated = x * cos_emb + self.rotate_quater(x) * sin_emb
        return x_rotated
    

class RotaryPositionEmbedding2DMix(nn.Module):
    def __init__(self, dim, num_heads=8,theta=10000,rotate=True,max_shape=(152,152)):
        super().__init__()
        self.dim = dim
        self.max_shape = max_shape
        
        h,w = max_shape

        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)# (2, heads,  dim//2)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)


    def forward(self, x):
        n,heads,d,h,w = x.shape
        t_x = torch.arange(w).to(x)
        t_y = torch.arange(h).to(x)
        # No float 16 for this range
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = (t_x.unsqueeze(-1) @ self.freqs[0].unsqueeze(-2)).unsqueeze(0).unsqueeze(2) # (1, heads, 1, w, d//2)
            freqs_y = (t_y.unsqueeze(-1) @ self.freqs[1].unsqueeze(-2)).unsqueeze(0).unsqueeze(3) # (1, heads, h, 1, d//2)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y) # heads, h, w, d//2
        
        x_comp = torch.view_as_complex(rearrange(x,'n heads (d v) h w -> n heads h w d v', v=2).contiguous())
        x_rotated = rearrange(torch.view_as_real(x_comp * freqs_cis).flatten(-2),
                             'n heads h w d -> n heads d h w')   
        return x_rotated
    
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'