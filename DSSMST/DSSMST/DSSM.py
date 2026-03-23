import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class DSSM(nn.Module):
    def __init__(self, dim, expand=2, state_dim=16, dt_rank=8, conv_kernel=3):
        super().__init__()
        self.dim = dim
        self.expand_dim = dim * expand
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        self.A = nn.Parameter(-torch.exp(torch.randn(self.expand_dim, state_dim))) 
        self.B = nn.Parameter(torch.randn(self.expand_dim, state_dim))
        self.C = nn.Parameter(torch.randn(state_dim, self.expand_dim))
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
        self.out_proj = nn.Linear(self.expand_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)

    def selective_scan(self, x):
        b, l, d = x.shape  # batch, length, dim
        dt = x.mean(dim=-1, keepdim=True).detach() 
        dt = self.dt_proj(dt)  # [b, l, expand_dim]

        m = torch.zeros(b, self.state_dim, device=x.device)  
        m_seq = []
        
        for t in range(l):
            u_t = x[:, t, :] 
            dt_t = dt[:, t, :].unsqueeze(-1)
            m = torch.einsum('dn,bd->bn', self.A * dt_t, m) + torch.einsum('dn,bd->bn', self.B * dt_t, u_t)
            m_seq.append(m)
        
        m_seq = torch.stack(m_seq, dim=1)  # [b, l, state_dim]
        y = torch.einsum('bld,nd->bln', m_seq, self.C) + self.D * x
        
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