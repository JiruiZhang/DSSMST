import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class DeterministicMamba(nn.Module):
    def __init__(self, dim, expand=2, state_dim=16, dt_rank=8, conv_kernel=3):
        super().__init__()
        self.dim = dim
        self.expand_dim = dim * expand
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        self.A = nn.Parameter(-torch.exp(torch.randn(self.expand_dim, state_dim)))
        self.D = nn.Parameter(torch.ones(self.expand_dim))
        
        self.dt_proj = nn.Linear(1, self.expand_dim, bias=False)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)

        self.conv = nn.Conv1d(
            in_channels=self.expand_dim,
            out_channels=self.expand_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel//2,
            groups=self.expand_dim,
            bias=False
        )
        nn.init.orthogonal_(self.conv.weight)

        self.in_proj = nn.Linear(dim, self.expand_dim * 2, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=1.0)

        self.BC_proj = nn.Linear(self.expand_dim, 2 * state_dim, bias=False)
        nn.init.normal_(self.BC_proj.weight, mean=0.0, std=0.02)
        
        self.out_proj = nn.Linear(self.expand_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)

    def selective_scan(self, x):

        dt = x.mean(dim=-1, keepdim=True).detach()  # [b, l, 1]
        dt = self.dt_proj(dt)  # [b, l, d]
        
        BC = self.BC_proj(x)  # [b, l, 2*n]
        B, C = BC.chunk(2, dim=-1)

        deltaA = torch.exp(torch.einsum('bld,dn->bldn', dt, self.A))
        deltaB_u = torch.einsum('bld,dn,bld->bldn', dt, self.A, x)

        h = torch.cumsum(deltaA * deltaB_u, dim=1) / \
            (torch.cumsum(deltaA, dim=1) + 1e-6)
        y = torch.einsum('bldn,dn->bld', h, self.A) + self.D * x
        
        return y

    def forward(self, x):
        residual = x

        x_conv, x_ssm = self.in_proj(x).chunk(2, dim=-1)

        x_conv = F.conv1d(
            x_conv.transpose(1, 2),
            self.conv.weight,
            padding=self.conv.padding[0],
            groups=self.expand_dim
        ).transpose(1, 2)

        x_ssm = self.selective_scan(x_ssm)

        gate = torch.sigmoid(x_ssm)
        x = x_conv * gate + x_ssm * (1 - gate)
        
        return self.out_proj(x) + residual