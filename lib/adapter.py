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

class AdapterEnsemble(nn.Module):
    """
    This layer uses only adapters and discards linear operation.
    It consists of two tensors. 
    - the input scaling  
    - the input bias    

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    """

    r: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        # ensemble_scaling_in: bool,
        # ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()
        hidden_dim = out_features

        # Have linear operation if this is the first adapter.
        if in_features != out_features:
            self.weight = nn.Parameter(torch.empty(hidden_dim, in_features))
        else:
            self.weight = None

        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(hidden_dim))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, hidden_dim))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.hidden_dim,
                dtype=self.r.dtype,
                device=self.r.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.hidden_dim)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # Apply scaling and and add bias.
        if self.r is not None:
            x = x * self.r
        if self.weight is not None:
            x = x @ self.weight.T
       
        if self.bias is not None:
            x = x + self.bias
        return x
