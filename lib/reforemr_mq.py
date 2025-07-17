import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiQueryAttention(nn.Module):
    """
    MultiQuery attention module.

    A variant of attention where each head has its own Query,
    but all heads share the same Key and Value (Multi-Query Attention).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_norm: bool = True,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        normalization: str = 'LayerNorm',
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias) if q_proj else None
        self.k = nn.Linear(dim, self.head_dim, bias=qkv_bias) if k_proj else None
        self.v = nn.Linear(dim, self.head_dim, bias=qkv_bias) if v_proj else None

        self.q_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = getattr(nn, normalization)(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()

    def reset_parameters(self):
       for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    def forward(
        self, 
        x_q: Tensor,              
        x_k: Tensor,              
        x_v: Tensor,              
    ):
        """
            x_q: (B, S, D)
            x_k: (B, N, D)
            x_v: (B, N, D)
        """
        if x_q.ndim == 2:
            x_q = x_q.unsqueeze(1)
        assert x_q.ndim == 3

        B, S, D = x_q.shape
        _, N, _ = x_k.shape

        q = self.q(x_q) if self.q is not None else x_q
        k = self.k(x_k) if self.k is not None else x_k # (B, N, D_H)
        v = self.v(x_v) if self.v is not None else x_v # (B, N, D_H)

        q = q.reshape(B, self.num_heads, -1, self.head_dim)  # (B, H, S, D_H)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale # d**-0.5

        attn = torch.einsum('bhsd,bnd->bhsn', q, k)
        attn = attn.softmax(dim=-1) # (B, H, S, N)
        attn = self.attn_drop(attn)
        x = torch.einsum('bhsn,bnd->bhsd', attn, v) 
        x = x.transpose(1, 2).reshape(B, S, D)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x