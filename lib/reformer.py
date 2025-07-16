import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """A dense module following attention in Transformer block."""
    
    def __init__(
        self,
        dim: int,
        dim_multiplier: float = 2.0,
        drop: float = 0.0,
    ):
        super(MLP, self).__init__()
        hidden_dim = int(dim * dim_multiplier)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.norm(x)
        x = self.drop2(self.act(self.fc2(x)))
        return x        

class Attention(nn.Module):
    """
        Single head attention module
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
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
        self.k = nn.Linear(dim, dim, bias=qkv_bias) if k_proj else None
        self.v = nn.Linear(dim, dim, bias=qkv_bias) if v_proj else None
        
        self.q_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = getattr(nn, normalization)(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.reset_parameters()

    def reset_parameters(self,):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_q, x_k, x_v):
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
        k = self.k(x_k) if self.k is not None else x_k
        v = self.v(x_v) if self.v is not None else x_v

        q = q.reshape(B, self.num_heads, -1, self.head_dim)  # (B, H, S, D_H)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, D_H)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, D_H)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale

        attn = torch.einsum('bhsd,bhnd->bhsn', q, k) # (B, H, S, N)
        attn = attn.softmax(dim=-1) # (B, H, S, N)
        attn = self.attn_drop(attn) # 
        x = attn @ v # (B, H, S, D_H)

        x = x.transpose(1, 2).reshape(B, S, D)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x # (B, S ,D)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_norm: bool = True,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        normalization: str = 'LayerNorm',
        dim_multiplier: int = 2,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.cross_attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            normalization=normalization,
            q_proj=False,
            k_proj=False,
            v_proj=False,
        )
        self.mlp = MLP(
            dim=dim,
            dim_multiplier=dim_multiplier,
            drop=mlp_drop,
        )

    def forward(self, x_q, x_k, x_v):
        if x_q.ndim == 2:
            x_q = x_q.unsqueeze(1)
        assert x_q.ndim == 3
        
        x_q = x_q + self.cross_attention(x_q, x_k, x_v)
        x_q = x_q + self.mlp(x_q)

        return x_q

class BaseEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_main: int,
        d_multiplier: float = 2.0,
        dropout_prob: float = 0.0,
        n_blocks: int = 1, 
        skip_connection: bool = False,
    ):
        super().__init__()
        d_hidden = int(d_main * d_multiplier)

        def make_block():
            return nn.Sequential(*[
                nn.Linear(d_main, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(d_hidden, d_main),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
        self.blocks = nn.ModuleList(
            [make_block() for _ in range(n_blocks)]
        )

        self.linear = nn.Linear(d_in, d_main) # first embedding layer
        self.skip_connection = skip_connection
        self.reset_parameters()

    def reset_parameters(self):
        
        pass

    def forward(self, x):
        x = self.linear(x) # d_in -> d_main
        for block in self.blocks:
            if self.skip_connection: 
                x = x + block(x)
            else: 
                x = block(x)
        return x
