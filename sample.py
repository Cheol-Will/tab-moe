import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Literal

import delu
import numpy as np
import rtdl_num_embeddings
import scipy
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

import lib
import lib.data
import lib.deep
import lib.env
from lib import KWArgs, PartKey

def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


class TopkRouter(nn.Module): 
    """
    Gating Network with logits
    input: (B, F)
    output: weights for all expert and top-k expert indices (B, num_experts) and (B, K)
    """
    def __init__(
        self,
        in_features: int,
        num_experts: int = 32,  
        k: int = 4,
    ):
        super(TopkRouter, self).__init__()
        self.route_linear = nn.Linear(in_features, num_experts, bias=False)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_rsqrt_uniform_(module.weight, module.in_features)
                if module.bias is not None:
                    init_rsqrt_uniform_(module.bias, module.in_features)

    def forward(self, x):
        logits = self.route_linear(x) # (B, D) -> (B, num_experts)

        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1) # (B, K)
        top_k_weights = F.softmax(top_k_logits, dim=-1) # (B, K)
        top_k_weights = top_k_weights.to(logits.dtype)

        full_weights = torch.zeros(
            logits.size(),
            dtype=logits.dtype,
            device=logits.device,
        )         
        full_weights.scatter_(1, top_k_indices, top_k_weights)

        return full_weights, top_k_indices # return weights and indices

class Expert(nn.Module):
    """
    Expert block with bottleneck.
    Lin -> ReLU -> Dropout -> Lin -> ReLU -> Dropout
    """

    def __init__(
        self,
        d_block: int,
        moe_ratio: float = 0.25,
        dropout: float = 0.0,
        activation: str = 'ReLU',
    ):
        super(Expert, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(d_block, int(d_block*moe_ratio)),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(int(d_block*moe_ratio), d_block),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
        )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_rsqrt_uniform_(module.weight, module.d_block)
                if module.bias is not None:
                    init_rsqrt_uniform_(module.bias, module.d_block)

    def forward(self, x):
        x = self.block(x)
        return x

class MoEBlcokEinSum(nn.Module):

    def __init__(
        self,
        d_block: int,
        moe_ratio: float = 0.25,
        dropout: float = 0.0,
        k: int = 4,
        num_experts: int = 32,
        activation: str = 'ReLU',
    ):
        super(MoEBlcokEinSum, self).__init__()
        self.router = TopkRouter(d_block, num_experts, k)
        
        self.weights1 = nn.Parameter(torch.empty(num_experts, d_block, int(d_block*moe_ratio)))
        self.act1 = getattr(nn, activation)()
        self.dropout1 = nn.Dropout(dropout)  
        self.weights2 = nn.Parameter(torch.empty(num_experts, int(d_block*moe_ratio), d_block))
        self.act2 = getattr(nn, activation)()
        self.dropout2 = nn.Dropout(dropout)  

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weights1, self.weights1.shape[-1])
        init_rsqrt_uniform_(self.weights2, self.weights2.shape[-1])

    def forward(self, x):
        weights, indices = self.router(x) # (B, E), (B, K)  

        x = torch.einsum("bd,edh->ebh", x, self.weights1) # (E, B, D)
        x = self.dropout1(self.act1(x))
        x = torch.einsum("ebh,ehd->ebd", x, self.weights2) # (E, B, D)
        x = self.dropout2(self.act2(x))
        x = x.transpose(0, 1) # (B, E, D)

        topk_x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1))) # (B, K, D)
        topk_weights = torch.gather(weights, 1, indices)  # (B, K)

        x = torch.einsum("bkd,bk->bd", topk_x, topk_weights)

        return x 
    
class MoEBlock(nn.Module):
    """
    Mixture of Expert Block with routing and experts.
    Each expert is a two-layer MLP.
    """
    def __init__(
        self,
        d_block: int,
        moe_ratio: float = 0.25,
        dropout: float = 0.0,
        k: int = 4,
        num_experts: int = 32,
        activation: str = 'ReLU',
    ):
        super(MoEBlock, self).__init__()
        self.router = TopkRouter(d_block, num_experts, k)
        self.experts = nn.ModuleList([
            Expert(d_block, moe_ratio, dropout, activation) for _ in range(num_experts)
        ])
        self.k = k
        self.num_experts = num_experts
        self.d_block = d_block

    def forward(self, x):
        assert x.ndim == 2
        weights, indices = self.router(x) # (B, num_experts), (B, K)
        out = torch.zeros(x.shape[0], self.d_block, dtype=x.dtype, device=x.device) # (B, D)

        for idx, expert in enumerate(self.experts):
            mask = (indices==idx).any(dim=-1) # Boolean mask: (B, )
            if mask.any():
                expert_input = x[mask] # (B_i, D)
                expert_output = expert(expert_input) # (B_i, D)
                scores = weights[mask, idx].unsqueeze(-1) # (B_i, 1)
                out[mask] += expert_output * scores # (B_i, D)

        # indices should be printed or saved for statistics.

        return out 

class MoEMLP(nn.Module):
    """
    MLP MOE
    """
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU', 
        moe_ratio: float = 0.25, 
        num_experts: int = 32,  
        k: int = 4,
    ) -> None:
        assert k > 0
        super(MoEMLP, self).__init__()
        d_in = d_block if d_in is None else d_in

        self.embed = nn.Linear(d_in, d_block)
        self.moe = nn.Sequential(*[
            MoEBlcokEinSum(
                d_block=d_block ,
                moe_ratio=moe_ratio,
                dropout=dropout,
                k=k,
                num_experts=num_experts,
                activation=activation,
            )
            for _ in range(n_blocks)
        ])
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

        self.d_in = d_in
        self.d_block = d_block
        self.num_experts = num_experts
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:

        init_rsqrt_uniform_(self.embed.weight, self.d_in)
        if self.embed.bias is not None:
            init_rsqrt_uniform_(self.embed.bias, self.d_in)

        # output
        if self.output is not None:
            init_rsqrt_uniform_(self.output.weight, self.d_block)
            if self.output.bias is not None:
                init_rsqrt_uniform_(self.output.bias, self.d_block)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x) # (B, F) -> (B, D)
        x = self.moe(x) # (B, D) -> (B, D)

        if self.output is not None:
            x = self.output(x) # (B, D) -> (B, d_out) if needed.

        return x


class SparseSharedMoE(nn.Module):
    """
    Sparse shared mixture of expert extends the SparseMoE 
    by including additional experts that are shared across all samples.
    """

    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out : None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "ReLU", 
        moe_ratio: float = 0.25, 
        num_experts: int = 32,
        k: int = 4,
    ) -> None:
        assert k > 0

        self.embed = nn.Linear(d_in, d_block)
        self.moe = nn.ModuleList(*[
            MoEBlcokEinSum(
                d_block=d_block ,
                moe_ratio=moe_ratio,
                dropout=dropout,
                k=k,
                num_experts=num_experts,
                activation=activation,
            )
            for _ in range(n_blocks)
        ])

        self.shared_expert = nn.ModuleList(*[
            nn.Sequential(
                nn.Linear(d_block, d_block),
                getattr(nn.activation)(),
                nn.Dropout(dropout),
                nn.Linear(d_block, d_block),
                getattr(nn.activation)(),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])

        self.output = None if d_out is None else nn.Linear(d_block, d_out)
        self.d_in = d_in
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.num_experts = num_experts
        self.k = k

    def reset_parameters(self) -> None:

        init_rsqrt_uniform_(self.embed.weight, self.d_in)
        if self.embed.bias is not None:
            init_rsqrt_uniform_(self.embed.bias, self.d_in)
        
        if self.output is not None:
            init_rsqrt_uniform_(self.output.weight, self.d_block)
            if self.output.bias is not None:
                init_rsqrt_uniform_(self.output.bias, self.d_block)

    def forward(self, x):
        x = self.embed(x) # (B, F) -> (B, D)
        for i in range(self.n_blocks):
            x = self.moe[i](x) + self.shared_expert[i](x) # (B, D)
        
        if self.output is not None:
            x = self.output(x) # (B, D) -> (B, d_out) if needed.

        return x