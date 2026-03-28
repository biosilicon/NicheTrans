from __future__ import absolute_import

import torch
from torch import nn

from model.attention import *
from model.nicheTrans import NetBlock


# NicheTrans with spatial information only
class NicheTrans(nn.Module):
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate

        self.fea_size, self.img_size = 256, 128

        ###############
        # omics encoder
        self.encoder = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)
    
        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)

        ##############
        self.predict_layers = nn.Sequential(nn.Linear(self.fea_size, 128),
                             nn.BatchNorm1d(128),
                             nn.LeakyReLU(),
                             nn.Linear(128, target_length, bias=False))

        ################
        # others
        self.non_linear = nn.Sequential(nn.Linear(256, 256),
                                        nn.LayerNorm(256),
                                        # nn.BatchNorm1d(9),
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
        b = source.size(0)
        l = source_neighbor.size(1)
        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, l//2, 1), self.token_neigh_2.repeat(1, l//2, 1)], dim=1)

        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder(omic_data).view(b, -1, self.fea_size) 
        f_omic = f_omic + spatial_tokens

        f_omic = self.non_linear(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        f = self.dropout(f_omic[:, 0, :])

        # final prediction
        out = self.predict_layers(f)

        return out
    