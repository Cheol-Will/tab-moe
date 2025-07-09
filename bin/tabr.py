import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Literal, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import rtdl_num_embeddings
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        bins: None | list[Tensor],
        #
        num_embeddings: None | dict = None,
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['dropout0']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[bool, Literal['dropout0']],
        normalization: str, # BatchNorm or LayerNorm
        activation: str,
        context_size: int = 96, # always same
        # Below options are used only if it's needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: None | int = None,
        # k: str | int = None, # dummy parameter
    ):
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        
        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features

        else:
            if bins is None:
                self.num_module = lib.deep.make_module(
                    **num_embeddings, n_features=n_num_features
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = lib.deep.make_module(**num_embeddings, bins=bins)

            # Adjust d_num for piecewise linear embedding and periodic embedding.    
            d_num = n_num_features * num_embeddings['d_embedding']
        
        # >>> Categorical features
        self.cat_module = (
            lib.deep.OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        d_cat = sum(cat_cardinalities)      

        # >>> Embedding
        d_in = d_num + d_cat
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )
        self.linear = nn.Linear(d_in, d_main) # first layer
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)] # First embedding has no LayerNorm. 
        )
        
        # >>> Retreival
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2)) # (B, 1, d_main) -> (B, d_main)
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout) # dropout for context 


        # >>> Prediction
        d_out = 1 if n_classes is None else n_classes
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, d_out),
        )
        self.context_size = context_size
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()


    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501
    
    def _encode(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x]) # (B, F)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))

        return x, k 

    def forward(
        self,
        *,
        x_num: None | Tensor = None, 
        x_cat: None | Tensor = None,
        y: None | Tensor = None, 
        candidate_x_num: None | Tensor = None, 
        candidate_x_cat: None | Tensor = None,
        candidate_y: None | Tensor = None,
        is_train: bool,
    ) -> Tensor:
        context_size = self.context_size
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(x_num=candidate_x_num, x_cat=candidate_x_cat)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(num_batch, cat_batch)[1]
                        for num_batch, cat_batch in zip(
                            delu.iter_batches(candidate_x_num, self.candidate_encoding_batch_size, shuffle=False),
                            delu.iter_batches(candidate_x_cat, self.candidate_encoding_batch_size, shuffle=False)
                        )
                    ]
                )
            )
        x, k = self._encode(x_num, x_cat)
        if is_train:
            # include current queries since they are excluded from candidate
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # The search below is optimizer for larger datasets.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            candidate_k = candidate_k.to(torch.float32)
            k = k.to(torch.float32)
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            self.search_index.reset()
            self.search_index.add(candidate_k)
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(
                k, context_size + (1 if is_train else 0) 
            ) # During training, query always searches itself; thus, search one more index
            
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1]) # discard inf-index                

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeat the same computation for the context objects and with autograd on.
            # concat -> indexing -> encode
            candidate_num = torch.cat([x_num, candidate_x_num])
            candidate_cat = torch.cat([x_cat, candidate_x_cat])
            context_k = self._encode(
                candidate_num[context_idx], candidate_cat[context_idx]
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # 
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        # print(f"[Debug] before context x: {x.shape}")

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>> prediction
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        # print(f"[Debug] x: {x.shape}")
        x = x[:, None] # (B, D_OUT) -> (B, 1, D_OUT)
        # print(f"[Debug] x: {x.shape}")

        return x

# def main(
#     config: Config | str | Path,
#     output: None | str | Path = None,
#     *,
#     force: bool = False,
# ) -> None | lib.JSONDict:
    
