import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import random
import os
import numpy as np
from .model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from .utils_func import *
import scipy.sparse as sp

def adj_to_edge_index(adj):
    if sp.issparse(adj):
        adj = adj.tocoo()
        row = adj.row
        col = adj.col
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        edge_index = torch.tensor([row, col], dtype=torch.long)
    else:
        edge_index = (adj != 0).nonzero().t().contiguous()
        edge_index, _ = torch.sort(edge_index, dim=1)
    return edge_index

class DSSMST():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        random_seed =2025,   
        alpha = 10, 
        beta = 1,   
        theta = 0.1, 
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        n_top_genes=2000
        ):
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha  
        self.beta = beta    
        self.theta = theta  
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
          
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata, n_top_genes=self.n_top_genes) 
        
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device) 
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device) 
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        self.edge_index = adj_to_edge_index(self.graph_neigh).to(self.device)

        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
           self.dim_input = self.feat_sc.shape[1] 
           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs

    def train(self):
        self.model = Encoder(self.dim_input,
                             self.dim_output,
                             self.graph_neigh,
                             dropout=0.3
                             ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        print('Begin to train ST data...')
        self.model.train().to(self.device)
        
        for epoch in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()
            self.features_a = self.features[torch.randperm(self.features.size(0))]
            h, ret, ret_a = self.model(feat=self.features,
                                       feat_a=self.features_a,
                                       adj=self.adj,
                                       edge_index=self.edge_index)
            loss_recon = self.model.calculate_reconstruction_loss(h, self.features) 
            loss_csl = self.model.calculate_contrastive_loss(ret, ret_a, self.label_CSL) 
            loss_total = self.alpha * loss_recon + self.beta * loss_csl

            loss_total.backward()
            self.optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Total Loss: {loss_total.item():.4f}, Recon Loss: {loss_recon.item():.4f}, CSL Loss: {loss_csl.item():.4f}")

        print("Optimization finished for ST data!")
        with torch.no_grad():
           self.model.eval() 
           h, ret, ret_a = self.model(feat=self.features,
                                      feat_a=self.features_a,
                                      adj=self.adj,
                                      edge_index=self.edge_index) 
           if self.deconvolution:
              self.emb_rec = h  
           else:
              if self.datatype in ['Stereo', 'Slide']:
                 self.emb_rec = F.normalize(h, p=2, dim=1).detach().cpu().numpy()
              else:
                 self.emb_rec = h.detach().cpu().numpy()
              self.adata.obsm['emb'] = self.emb_rec
           return self.adata

    def loss(self, emb_sp, emb_sc):
        map_probs = F.softmax(self.map_matrix, dim=1)
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        ave = torch.div(p, k) 
        loss = - torch.log(ave).mean()
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + 1e-12  
        M = torch.div(M, Norm)
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)
        return M