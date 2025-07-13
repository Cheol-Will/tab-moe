import itertools
from typing import Any, Literal
import math

import delu
import faiss
import faiss.contrib.torch_utils  # << this line makes faiss work with PyTorch
import rtdl_num_embeddings
import rtdl_revisiting_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)

class ParallelLayerNorm(nn.Module):
    """
    Apply k parallel layer normalizaion layers.

    """

    def __init__(
        self,
        hidden_dim: int,
        k: int = 4, 
        eps: float = 1e-5,
        bias: bool = True,
        *,
        init: Literal['ones', 'normal', 'random-signs'] = 'ones',
        device=None,
        dtype=None,
    ) -> None:
        """
        hidden_dim: hidden dimension
        k: the number of parallel layer norms
        """
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k
        self.eps = eps
        self._weight_init = init

        self.weight = nn.Parameter(
            torch.empty((k, hidden_dim), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((k, hidden_dim), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')
        
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0, std=1/self.hidden_dim**0.5)

    def forward(self, x: Tensor) -> Tensor:
        """
        input: (B, K, D)
        output: (B, K, D)
        """
        assert x.ndim == 3
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        
        x = x * self.weight.unsqueeze(0) 
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return x


class ScalePLNEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.pln = ParallelLayerNorm(hidden_dim=d, k=k, init=init)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return self.pln(x)


class ParallelLayerNormEnsemble(nn.Module):
    """
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.pln = ParallelLayerNorm(
            hidden_dim=in_features,
            k=k,
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        x = self.pln(x)
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x