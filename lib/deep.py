import itertools
from typing import Any, Literal

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


# ======================================================================================
# Modules
# ======================================================================================
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class NLinear(nn.Module):
    """A stack of N linear layers. Each layer is applied to its own part of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PiecewiseLinearEmbeddingsV2(rtdl_num_embeddings.PiecewiseLinearEmbeddings):
    """
    This class simply adds the default values for `activation` and `version`.
    """

    def __init__(
        self,
        *args,
        activation: bool = False,
        version: None | Literal['A', 'B'] = 'B',
        **kwargs,
    ) -> None:
        # Future Work you ignored version since error says that
        # PiecewiseLinearEmbeddings does not get kwarg version.
         
        # super().__init__(*args, **kwargs, activation=activation, version=version)
        super().__init__(*args, **kwargs, activation=activation)


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
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

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
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
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
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
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
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

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


def make_efficient_ensemble(module: nn.Module, EnsembleLayer, **kwargs) -> None:
    """Replace linear layers with efficient ensembles of linear layers.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                EnsembleLayer(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)


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
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_rsqrt_uniform_(module.weight, module.in_features)
                if module.bias is not None:
                    init_rsqrt_uniform_(module.bias, module.in_features)
    
    def forward(self, x):
        x = self.block(x)
        return x


class MoEBlockEinSum(nn.Module):

    def __init__(
        self,
        d_block: int,
        moe_ratio: float = 0.25,
        dropout: float = 0.0,
        k: int = 4,
        num_experts: int = 32,
        activation: str = 'ReLU',
    ):
        super(MoEBlockEinSum, self).__init__()
        self.router = TopkRouter(d_block, num_experts, k)
        
        self.weights1 = nn.Parameter(torch.empty(num_experts, d_block, int(d_block*moe_ratio)))
        # self.bias1 = nn.Parameter(torch.empty(num_experts, int(d_block*moe_ratio)))
        self.act1 = getattr(nn, activation)()
        self.dropout1 = nn.Dropout(dropout)  
        self.weights2 = nn.Parameter(torch.empty(num_experts, int(d_block*moe_ratio), d_block))
        # self.bias2 = nn.Parameter(torch.empty(num_experts, d_block))
        self.act2 = getattr(nn, activation)()
        self.dropout2 = nn.Dropout(dropout)  

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weights1, self.weights1.shape[-1])
        init_rsqrt_uniform_(self.weights2, self.weights2.shape[-1])

        # init_rsqrt_uniform_(self.bias1, self.weights1.shape[-1])
        # init_rsqrt_uniform_(self.bias2, self.weights2.shape[-1])

    def forward(self, x, return_route = False):
        weights, indices = self.router(x) # (B, E), (B, K)  
        # batch_size, hidden_dim = x.shape

        x = torch.einsum("bd,edh->ebh", x, self.weights1) # (E, B, D)
        # x = x + self.bias1.unsqueeze(1).expand(-1, batch_size, -1) # (E, B, D)
        x = self.dropout1(self.act1(x))
        x = torch.einsum("ebh,ehd->ebd", x, self.weights2) # (E, B, D)
        # x = x + self.bias2.unsqueeze(1).expand(-1, batch_size, -1) # (E, B, D)
        x = self.dropout2(self.act2(x))
        x = x.transpose(0, 1) # (B, E, D)

        # extract 
        topk_x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1))) # (B, K, D)
        topk_weights = torch.gather(weights, 1, indices)  # (B, K)

        x = torch.einsum("bkd,bk->bd", topk_x, topk_weights)

        if return_route:
            return x, indices # return routing if needed
        else:
            return x 
    

class MoESparse(nn.Module):
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
        super(MoESparse, self).__init__()
        d_in = d_block if d_in is None else d_in

        self.embed = nn.Linear(d_in, d_block)
        self.moe = nn.ModuleList([
            MoEBlockEinSum(
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
        self.n_blocks = n_blocks
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

    def forward(self, x: Tensor, return_route = False) -> Tensor:
        x = self.embed(x) # (B, F) -> (B, D)
        route = None
        for i in range(self.n_blocks):
            if (i == 0) and return_route:
                x, route = self.moe[i](x, return_route) # (B, D)
            else:
                # perform moe + shared expert for the rest of the blocks.
                x = self.moe[i](x) # (B, D)
        
        if self.output is not None:
            x = self.output(x) # (B, D) -> (B, d_out) if needed.

        # return the first routing result if needed.
        return (x, route) if return_route else x


class MoESparseShared(MoESparse):
    """
    Sparse shared mixture of expert extends the SparseMoE 
    by including additional experts that are shared across all samples.
    """

    def __init__(
        self,
        *,
        d_in: int | None = None,
        d_out: int | None = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "ReLU",
        moe_ratio: float = 0.25,
        num_experts: int = 32,
        k: int = 4,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            activation=activation,
            moe_ratio=moe_ratio,
            num_experts=num_experts,
            k=k,
        )

        # only add shared expert
        self.shared_expert = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_block, d_block),
                getattr(nn, activation)(),
                nn.Dropout(dropout),
                nn.Linear(d_block, d_block),
                getattr(nn, activation)(),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])

        self.reset_shared_expert_parameters()

    def reset_shared_expert_parameters(self):
        for seq in self.shared_expert:
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    init_rsqrt_uniform_(layer.weight, layer.in_features)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    def forward(self, x: Tensor, return_route: bool = False):
        x = self.embed(x)  # (B, D)
        route = None

        for i in range(self.n_blocks):
            if i == 0 and return_route:
                out, route = self.moe[i](x, return_route)  # (B, D), indices
            else:
                out = self.moe[i](x)                        # (B, D)
            x = out + self.shared_expert[i](x)             # (B, D)

        if self.output is not None:
            x = self.output(x)   # (B, d_out)

        # return the first routing result if needed.
        return (x, route) if return_route else x


class MLP_Block(nn.Module):
    """
    ModernNCA style MLP block.
    """

    def __init__(
        self, 
        d_in: int, 
        d_block: int, 
        dropout: float, 
        activation: str = "ReLU",
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d_block),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(d_block, d_in)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_rsqrt_uniform_(module.weight, module.in_features)
                if module.bias is not None:
                    init_rsqrt_uniform_(module.bias, module.in_features)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
import faiss
import faiss.contrib.torch_utils  # << this line makes faiss work with PyTorch

class TabRM(nn.Module):
    """
    Retrieve top-k neighbors and create k different views of query.
    Generate k-multiple predictions. 
    """
    
    def __init__(
        self,
        *,
        d_in: int | None = None,
        d_out: int | None = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "ReLU",
        k: int = 32,
        # sample_rate: float = 0.8,
        memory_efficient: bool = True,
    ) -> None:
        super(TabRM, self).__init__()
        d_in = d_block if d_in is None else d_in

        # embedding process: 
        # Linear -> BatchNorm -> Linear -> ReLU -> Drop -> Linear -> BatchNorm
        self.embed = nn.Sequential(*[
            nn.Linear(d_in, d_block),
            MLP_Block(d_block, d_block, dropout, activation), # no bottleneck or expansion in embedding MLP
            nn.BatchNorm1d(d_block),
        ])

        # [linear -> ReLU -> Dropout] x n_blocks
        # Since TabRM forms K separate views by concatenating the query with each retrieved key vector
        # hidden dimension is 2*d_block
        self.mlp = MLP(
            d_in=2*d_block, 
            n_blocks=n_blocks, 
            d_block=2*d_block, 
            dropout=dropout, 
            activation=activation
        )

        self.output = None if d_out is None else nn.Linear(2*d_block, d_out)

        self.d_in = d_in
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.k = k
        # self.sample_rate = sample_rate # for subset sampling during trainig
        self.search_index = None
        self.memory_efficient = memory_efficient

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init embedding layer
        for module in self.embed:
            if isinstance(module, nn.Linear):
                init_rsqrt_uniform_(module.weight, module.in_features)
                if module.bias is not None:
                    init_rsqrt_uniform_(module.bias, module.in_features)

        # init MLPs
        for block in self.mlp.blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    init_rsqrt_uniform_(layer.weight, layer.in_features)
                    if layer.bias is not None:
                        init_rsqrt_uniform_(layer.bias, layer.in_features)

        # below lines are not likely to be used.
        if self.mlp.output is not None:
            init_rsqrt_uniform_(self.mlp.output.weight, self.mlp.output.in_features)
            if self.mlp.output.bias is not None:
                init_rsqrt_uniform_(self.mlp.output.bias, self.mlp.output.in_features)

        if self.output is not None:
            init_rsqrt_uniform_(self.output.weight, self.output.in_features)
            if self.output.bias is not None:
                init_rsqrt_uniform_(self.output.bias, self.output.in_features)

    # @torch.no_grad()
    def retrieve(self, x, candidate_x):
        """
        Retrieve from candidate_x.
        During training, candidate_x is a subset of train dataset 
            for computational efficieny and stochasticity. 
        """

        B, F = x.shape
        N, _ = candidate_x.shape
        
        # print(f"[Debug]: retrieve batch-size B={B}, pool-size N={N}, k={self.k}")
        # if N == 0:
        #     print(self.training)

        x = self.embed(x) # (B, F) -> (B, D)
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_xn = self.embed(candidate_x) 

        batch_size, d_block = x.shape
        with torch.no_grad():
            candidate_xn_ = candidate_xn.to(torch.float32)
            x_ = x.to(torch.float32) # need to convert to float 32 since the benchmark uses AMP with FP16

            if self.search_index is None:
                self.search_index = (
                    (
                        faiss.GpuIndexFlatL2 # defulat L2 distance
                    )(faiss.StandardGpuResources(), d_block)
                    if x.device.type == 'cuda'
                    else (faiss.IndexFlatL2)(d_block)
                )
            self.search_index.reset()
            self.search_index.add(candidate_xn_)
            context_idx: Tensor
            _, context_idx = self.search_index.search(
                x_, self.k
            )
        
        if self.memory_efficient and torch.is_grad_enabled():
            # the gradient of embedding of context are not being tracked.
            # thus, calculate the embedding once again to track gradient.   
            B, K = batch_size, self.k
            context_x = candidate_x[context_idx.reshape(-1)] # (B*K)
            context_x = self.embed(context_x).reshape(B, K, -1) 
        else:
            # the gradient of embedding of context are being tracked.   
            context_x = candidate_xn[context_idx] # retrieve 

        return x, context_x # (B, D), (B, K, D)

    def forward(self, x: Tensor, candidate_x) -> Tensor:
        """
            inputs: 
            x: tensor (B, F)
            candidate_x: training dataset (N, F)
            During training, query itself is not included in candidate_x. 
                        
            output: tensor (B, K, D)
            Note that output is k-multiple prediction;
              thus, successive may use NLINEAR.
        """

        # return embeded query and keys 
        x, context_x = self.retrieve(x, candidate_x) # (B, D), (B, K, D) 
        
        x = x.unsqueeze(1).expand(-1, self.k, -1) # (B, K, D)
        x = torch.cat([x, context_x], dim=-1)  # (B, K, 2D)
        x = self.mlp(x) # (B, K, 2D)
        if self.output is not None:
            x = self.output(x)   # (B, d_out)

        # concat query x and keys -> (B, K, D), which is k different views of query.
        # Keep in mind that self.output should be NLINEAR since this class outputs k-multiple predictions

        return x

class TabRMv2(TabRM):
    """
    Retrieve top-k neighbors and create k different views of query.
    Generate k-multiple predictions via batch-ensemble.
    """    
    def __init__(
        self,
        *,
        d_in: int | None = None,
        d_out: int | None = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "ReLU",
        k: int = 32,
        # sample_rate: float = 0.8,
        memory_efficient: bool = True,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            activation=activation,
            k=k,
            memory_efficient=memory_efficient,
        )
        self.mlp = nn.Sequential(*[
            LinearEfficientEnsemble(
                in_features=2*d_block,
                out_features=2*d_block,
                bias=True,
                k=k,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=True,
                scaling_init='ones' # initialize scale parameters (adapter) with 1.
            )
            for _ in range(n_blocks)
        ])


class TabRMv2Mini(TabRM):
    """
    Retrieve top-k neighbors and create k different views of query.
    Apply adapter (scaling) to k queries and shared MLPs.
    Note that it still uses k separate output layers.
    """    
    def __init__(
        self,
        *,
        d_in: int | None = None,
        d_out: int | None = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "ReLU",
        k: int = 32,
        # sample_rate: float = 0.8,
        memory_efficient: bool = True,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            activation=activation,
            k=k,
            memory_efficient=memory_efficient,
        )

        self.mlp = nn.Sequential(*[
            ScaleEnsemble(
                k,
                2*d_block,
                init='random-signs',
            ),
            MLP(
            d_in=2*d_block, 
            n_blocks=n_blocks, 
            d_block=2*d_block, 
            dropout=dropout, 
            activation=activation
            )
        ])



_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        PiecewiseLinearEmbeddingsV2,
        MLP,
        MoESparseShared, # you can remove 
        MoESparse,
    ]
}


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_revisiting_models.LinearEmbeddings
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    Optimizer = getattr(torch.optim, type)
    return Optimizer(**kwargs)
