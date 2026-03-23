import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv  
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from.preprocess import fix_seed
from DSSM import DSSM
from torch_geometric.nn import MessagePassing
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        if mask is None:
            mask = torch.eye(emb.size(0)).to(emb.device)
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1, keepdim=True) + 1e-8 
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)   

class DSSM(MessagePassing):
    def __init__(self, in_features, mamba_dim):
        super().__init__() 

        self.mamba_fc = nn.Linear(in_features, mamba_dim)
        self.norm1 = nn.LayerNorm(mamba_dim, eps=1e-6) 
        self.mamba = DSSM(dim=mamba_dim)
        self.residual_layer = nn.Linear(in_features, mamba_dim)
        self.norm2 = nn.LayerNorm(mamba_dim)

    def reset_parameters(self):
        super().reset_parameters()
        self.mamba_fc.reset_parameters()
        self.residual_layer.reset_parameters()

    def forward(self, x):
        residual = self.residual_layer(x)
        mamba_out = self.mamba(x.unsqueeze(0)).squeeze(0)
        out = self.norm2(mamba_out + residual) 

        return out


class DSSMs(nn.Module):
    def __init__(self, in_features, out_features,dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mamba = DSSM(self.in_features, self.out_features)
        self.norm = nn.LayerNorm(self.out_features)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        super().reset_parameters()
        self.norm.reset_parameters()


    def forward(self, feat):
        h = feat
        h = self.mamba(h)  
        h = self.norm(h)
        h = self.dropout(h)   
        return h

class Encoder(nn.Module):
    def __init__(self, in_features, out_features, graph_neigh,dropout=0.0,act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.graph_neigh = graph_neigh
        self.dssm = DSSMs(self.out_features, self.out_features,dropout=self.dropout)

        self.act = act

        self.nn1 = nn.Linear(self.out_features, self.out_features)
        self.nn2 = nn.Linear(self.in_features,self.in_features)
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features)) 
        self.reset_parameters() 
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):

        z = F.dropout(feat, self.dropout, self.training)  
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        z = self.nn1(z)
    
        z = self.dssm(z) 
        
        h = torch.mm(z, self.weight2)
        h = self.nn2(h)
        h = torch.mm(adj, h)  
        
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        z_a = self.nn1(z_a) 
        z_a = self.dssm(z_a)
        
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh) 
        g_a = self.sigm(g_a) 

        ret = self.disc(g, emb, emb_a) 
        ret_a = self.disc(g_a, emb_a, emb) 
        return h, ret, ret_a

