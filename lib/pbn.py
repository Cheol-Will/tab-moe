import torch
import torch.nn as nn

class ParallelBatchNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        k: int,
        eps: float = 1e-5,
        momentum: float = 0.9,
        init_type: str = 'xavier',
        gain: float = 1.0,
    ):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(k, hidden_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(k, hidden_dim), requires_grad=True)

        self.running_mean = nn.Parameter(torch.Tensor(k, hidden_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.Tensor(k, hidden_dim), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.Tensor(k, 1), requires_grad=False)

        self.hidden_dim = hidden_dim
        self.k = k
        self.eps = eps
        self.momentum = momentum
        self.init_type = init_type
        self.gain = gain
        self.reset_parameters()


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        # Initialize running statistics.
        self.reset_running_stats()

        # Initialize gamma and beta
        if self.init_type == 'bernoulli':
            with torch.no_grad():
                gamma_init = torch.ones(self.k, self.hidden_dim)
                beta_int = torch.ones(self.k, self.hidden_dim)
                gamma_init.bernoulli_(0.5).mul_(2).add_(-1).mul_(self.gain)
                beta_int.bernoulli_(0.5).mul_(2).add_(-1).mul_(self.gain)
                self.gamma = nn.Parameter(gamma_init, requires_grad=True)
                self.beta = nn.Parameter(beta_int, requires_grad=True)
        elif self.init_type == 'xavier':
            nn.init.xavier_uniform_(self.gamma, gain=self.gain)
            nn.init.xavier_uniform_(self.beta, gain=self.gain)
        else:
            print('WARNING: Wrong init type - PBNs are not initilized.')

    def batch_norm(self, input, running):
        pass


    def forward(self, x):
        assert x.ndim == 3

