from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

class AdapterMoE(nn.Module):
    """
    Route into separated adapter or shared linear layer.
        
    NOTE
    If this is the first layer in the model, 
    then it applies adapter and linear operation without routing.
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

        if in_features != out_features:
            self.router = None
        else:
            self.router = nn.Parameter(torch.empty(k, 2, in_features))

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
            ),  # type: ignore[code]
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
        if self.router is not None:
            init_rsqrt_uniform_(self.router, self.in_features)
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
                self.out_features,
                dtype=self.r.dtype,
                device=self.r.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.out_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # Apply scaling and and add bias.
        if self.router is not None:
            B, _, _ = x.shape
            logits = torch.einsum('bkd,kid->bki', x, self.router) 
            x_adapter, x_linear = None, None
            # Adapter
            if self.r is not None:
                x_adapter = x * self.r
            if self.bias is not None:
                x_adapter = x_adapter + self.bias # (B, K, D)
            # Linear
            if self.weight is not None:
                x_linear = x @ self.weight.T # (B, K, D)

            # Masking
            prob = F.softmax(logits, dim=-1) # (B, K, 2)
            x = x_adapter * prob[:, :, 0].unsqueeze(-1) + x_linear * prob[:, :, 1].unsqueeze(-1)
            # mask = torch.argmax(logits, dim=-1).unsqueeze(-1).float() # (B, K, 1)
            # x = x_adapter * mask + x_linear * (1 - mask) # 1 for adapter
        else:
            if self.r is not None:
                x = x * self.r
            if self.weight is not None:
                x = x @ self.weight.T
            if self.bias is not None:
                x = x + self.bias

        return x