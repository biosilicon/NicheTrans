from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *


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
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1,
                 n_spot_types=1):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.n_spot_types = n_spot_types

        self.fea_size, self.img_size = 256, 128

        ###############
        # omics encoder
        self.encoder = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)

        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)

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
        self.token_neigh_1 = nn.Parameter(torch.randn((self.n_spot_types, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((self.n_spot_types, self.fea_size), requires_grad=True))
        self.token_center_emb = nn.Embedding(n_spot_types, self.fea_size)

        trunc_normal_(self.token_center_emb.weight, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)

    def _expand_legacy_neighborhood_token(self, token):
        if token.shape == (self.n_spot_types, self.fea_size):
            return token
        if token.ndim == 3 and token.shape == (1, 1, self.fea_size):
            token = token.view(1, self.fea_size)
        if token.ndim == 2 and token.shape == (1, self.fea_size):
            return token.expand(self.n_spot_types, -1).clone()
        return token

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for token_name in ('token_neigh_1', 'token_neigh_2'):
            key = prefix + token_name
            if key in state_dict:
                state_dict[key] = self._expand_legacy_neighborhood_token(state_dict[key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _get_neighborhood_tokens(self, spot_type, ring_length):
        neigh1 = self.token_neigh_1[spot_type].unsqueeze(1).expand(-1, ring_length, -1)
        neigh2 = self.token_neigh_2[spot_type].unsqueeze(1).expand(-1, ring_length, -1)
        return neigh1, neigh2

    def forward(self, data, spot_type=None):
        b = data.size(0)
        l = data.size(1) - 1
        if spot_type is None:
            spot_type = torch.zeros(b, dtype=torch.long, device=data.device)
        else:
            spot_type = spot_type.long()
        center_token = self.token_center_emb(spot_type).unsqueeze(1)  # (B, 1, fea_size)
        neigh1_tokens, neigh2_tokens = self._get_neighborhood_tokens(spot_type, l // 2)
        spatial_tokens = torch.cat([center_token,
                                    neigh1_tokens,
                                    neigh2_tokens], dim=1)

        omic_data = data.view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder(omic_data).view(b, -1, self.fea_size) 
        f_omic = f_omic + spatial_tokens

        f_omic = self.non_linear(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        f = self.dropout(f_omic[:, 0, :])

        # final prediction
        # out = []
        # for i in range(self.target_length):
        #     out.append(self.predict_layers[i](f))
        # out = torch.cat(out, dim=1)

        out = self.predict_layers[25](f)

        return out
    
