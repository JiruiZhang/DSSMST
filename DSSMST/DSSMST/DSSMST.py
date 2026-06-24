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
import torch.optim as optim
import scipy.sparse as sp

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
        gamma = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        n_top_genes=2000
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
          
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata,)
        
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
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
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
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs

        coords = self.adata.obsm['spatial']
        score = coords[:, 0] + coords[:, 1]  
        order = np.argsort(score)
        self.spatial_order = torch.LongTensor(order).to(self.device)
        
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(score))
        self.spatial_inverse_order = torch.LongTensor(inverse_order).to(self.device)

    def train(self):

        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder(self.dim_input,
                                 self.dim_output,
                                 self.graph_neigh,
                                 dropout=0.3
                                 ).to(self.device)
        else:
            self.model = Encoder(self.dim_input,
                                 self.dim_output,
                                 self.graph_neigh,
                                 dropout=0.3
                                 ).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        # 优化器
        print('Begin to train ST data...')
        self.model.train().to(self.device)

        for epoch in tqdm(range(self.epochs)):

            self.model.train()
            self.features_a = self.features[torch.randperm (self.features.size (0))] 

            h, emb, ret, ret_a  = self.model(feat=self.features,
                                                          feat_a=self.features_a,
                                                          adj = self.adj,
                                                          spatial_order=self.spatial_order,            
                                                          spatial_inverse_order=self.spatial_inverse_order
                                                          )
 
            loss_sl_1 = self.loss_CSL(ret, self.label_CSL.to(self.device))
            loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL.to(self.device))

            loss_feat = F.mse_loss(self.features, h)
            loss_spatial = self.Noise_Cross_Entropy(emb, emb) 

            loss  = self.alpha*loss_feat+ self.beta*(loss_sl_1 + loss_sl_2) + self.gamma*loss_spatial


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished for ST data!")

        with torch.no_grad():
           self.model.eval() 
           h, emb, ret, ret_a = self.model(feat=self.features,
                                feat_a=self.features_a,
                                adj=self.adj,
                                spatial_order=self.spatial_order,            
                                spatial_inverse_order=self.spatial_inverse_order)
           if self.deconvolution:
              self.emb_rec = emb  
           else:
              if self.datatype in ['Stereo', 'Slide']:
                 self.emb_rec = F.normalize(emb, p=2, dim=1).detach().cpu().numpy()
              else:
                 self.emb_rec = emb.detach().cpu().numpy()
              self.adata.obsm['emb'] = self.emb_rec

           return self.adata # or self.emb_rec if deconvolution
        
    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k) 
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M) #可以指定为0或者batch的平均值

        return M