import torch
import torch
import torch.nn as nn

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
    
    def forward(self, x):
        
        return x

class Expert(nn.Module): 
    def __init__(self, ):
        pass
    
    def forward(self, x):

        return x


# SparseMoE class for the sparse mixture of experts module
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # input: (B, D)
        gating_output, indices = self.router(x) # (B, )
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class MLPExpert(nn.Module):
    """하나의 Expert로 작동할 MLP."""
    def __init__(self, *, d_in: int, n_blocks: int, d_block: int, dropout: float):
        super().__init__()
        # MLP와 동일하게 n_blocks개의 block을 쌓습니다.
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(d_block if i>0 else d_in, d_block),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(n_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)  # (B, d_block)
    

class MoE_MLP(nn.Module):
    """Top-k Router + MLPExpert 리스트를 조합한 Mixture-of-Experts."""
    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        num_experts: int,
        k: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
    ):
        super().__init__()
        # 1) 라우터: 입력 D_in → num_experts logits
        self.router = TopkRouter(d_in, num_experts, k)
        # 2) 전문가들: 모두 같은 MLP 구조
        self.experts = nn.ModuleList([
            MLPExpert(d_in=d_in, n_blocks=n_blocks, d_block=d_block, dropout=dropout)
            for _ in range(num_experts)
        ])
        # 3) 최종 선형: d_block → d_out
        self.output = nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d_in)
        weights, indices = self.router(x)        # weights: (B, E), indices: (B, k)
        # 각 배치별로 top-k 전문가만 실행하고 누적
        outs = x.new_zeros(x.size(0), self.output.out_features)  # (B, d_out)
        for ei, expert in enumerate(self.experts):
            mask = (indices == ei).any(dim=-1)  # (B,)
            if not mask.any():
                continue
            x_sel = x[mask]                    # (B_i, d_in)
            h = expert(x_sel)                  # (B_i, d_block)
            h = self.output(h)                 # (B_i, d_out)
            w = weights[mask, ei].unsqueeze(1) # (B_i, 1)
            outs[mask] += h * w                # weighted sum
        return outs  # (B, d_out)



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

