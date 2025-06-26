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


class ModelMoE(nn.Module):
    """MoE"""
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        backbone: dict,
        bins: None | list[Tensor],  # For piecewise-linear encoding/embeddings.
        num_embeddings: None | dict = None,
        arch_type: str = "moe-mlp",
        k: None | int  = None,
    ) -> None:
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        super().__init__()

            # >>> Continuous (numerical) features
        first_adapter_sections = []  # See the comment in `_init_first_adapter`.

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        else:
            if bins is None:
                self.num_module = lib.deep.make_module(
                    **num_embeddings, n_features=n_num_features
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = lib.deep.make_module(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings['d_embedding']
            first_adapter_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        # >>> Categorical features
        self.cat_module = (
            lib.deep.OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        first_adapter_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        d_flat = d_num + d_cat
        self.minimal_ensemble_adapter = None

        print(f"Initiailize backbone as {arch_type}")
        self.backbone = lib.deep.MoEMLP(d_in=d_flat, **backbone)

        # >>> Output
        d_block = backbone['d_block']
        d_out = 1 if n_classes is None else n_classes
        self.output = (
            nn.Linear(d_block, d_out)
            if arch_type in ['plain', 'moe-mlp']
            else lib.deep.NLinear(k, d_block, d_out)  # type: ignore[code]
        )

        # >>>
        self.arch_type = arch_type
        self.k = k


    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        # preprocess
        x = []
        
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x]) # (B, F)
        
        x = self.backbone(x) # (B, K, F) -> (B, K, D) or (B, F) -> (B, D)
        x = self.output(x) # (B, 1, D_OUT) or (B, D_OUT)
        x = x[:, None] # (B, D_OUT) -> (B, 1, D_OUT)

        return x        