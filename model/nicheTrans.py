from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.gnn_niche_encoder import GNNNicheEncoder


class NetBlock(nn.Module):
    def __init__(self, nlayer: int, dim_list: list, dropout_rate: float, noise_rate: float):

        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(nn.LeakyReLU())
            if not i == nlayer -1: 
                self.dropout_list.append(nn.Dropout(dropout_rate))
        
    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer -1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)

        return x


# NicheTrans with spatial information only
class NicheTrans(nn.Module):
    def __init__(
        self,
        source_length=877,
        target_length=137,
        noise_rate=0.2,
        dropout_rate=0.1,
        # --- context encoder switch ---
        context_model_type='gnn',   # 'transformer' | 'gnn'
        gnn_num_layers=2,                   # GNN: number of GraphSAGE layers
        gnn_hidden_dim=None,                # GNN: hidden dim (None → use fea_size)
        gnn_graph_type='full',              # GNN: 'full' (all-to-all) | 'star'
    ):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.context_model_type = context_model_type

        self.fea_size, self.img_size = 256, 128

        ###############
        # omics encoder
        self.encoder = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        ###############
        # context / neighborhood encoder (Transformer OR GNN)
        if context_model_type == 'transformer':
            self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
            self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)
            self.ln1 = nn.LayerNorm(self.fea_size)
            self.ln2 = nn.LayerNorm(self.fea_size)
        elif context_model_type == 'gnn':
            # GNN encoder: same in/out shape as the Transformer block above.
            # in_dim = fea_size (output of omics encoder + spatial tokens)
            # out[:, 0, :] is used for prediction (center node, index 0).
            self.gnn_encoder = GNNNicheEncoder(
                in_dim=self.fea_size,
                hidden_dim=gnn_hidden_dim or self.fea_size,
                num_layers=gnn_num_layers,
                dropout=self.dropout_rate,
                graph_type=gnn_graph_type,
            )
        else:
            raise ValueError(
                f"context_model_type must be 'transformer' or 'gnn', "
                f"got '{context_model_type}'"
            )

        ##############
        # prediction layers
        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(nn.Linear(self.fea_size, 128),
                             nn.BatchNorm1d(128),
                             nn.LeakyReLU(),
                             nn.Linear(128, 1, bias=True))
            )
        self.predict_layers = nn.ModuleList(predict_net)

        ################
        # others
        self.non_linear = nn.Sequential(nn.Linear(256, 256),
                                        nn.LayerNorm(256),
                                        nn.LeakyReLU())
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_5 = nn.Dropout(0.5)

        ################
        # initialize tokens for semantic embedding
        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))

        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)


    def forward(self, source, source_neighbor):
        # source:          [b, source_length]       — center cell omics
        # source_neighbor: [b, l, source_length]    — l neighbor omics
        b = source.size(0)
        l = source_neighbor.size(1)

        # Learnable spatial role tokens: [1, 1+l, fea_size]
        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, l//2, 1), self.token_neigh_2.repeat(1, l//2, 1)], dim=1)

        # Stack center + neighbors, flatten for shared encoder.
        # Shape: [b*(1+l), source_length]
        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # Encode each token independently: [b, 1+l, fea_size]
        f_omic = self.encoder(omic_data).view(b, -1, self.fea_size)
        # Inject spatial role information.
        f_omic = f_omic + spatial_tokens

        # Non-linear projection (shared by both paths).
        f_omic = self.non_linear(f_omic)

        # --- Context encoder: Transformer or GNN ---
        if self.context_model_type == 'transformer':
            # Original Transformer path (self-attention + FFN with residuals).
            # f_omic: [b, 1+l, fea_size]
            f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
            f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic
        else:  # 'gnn'
            # GNN path: build local niche graph, run GraphSAGE message passing.
            # f_omic: [b, 1+l, fea_size]  →  [b, 1+l, hidden_dim]
            # Node 0 is still the center cell (convention preserved).
            f_omic = self.gnn_encoder(f_omic)

        # Extract center-node representation (index 0, same as original).
        # f_omic[:, 0, :]: [b, fea_size]
        f = self.dropout(f_omic[:, 0, :])

        # Per-target prediction heads.
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        # Output: [b, target_length]
        out = torch.cat(out, dim=1)

        return out

