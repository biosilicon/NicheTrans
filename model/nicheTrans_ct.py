from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *


class NetBlock(nn.Module):
    def __init__(
        self,
        nlayer: int,
        dim_list: list,
        dropout_rate: float,
        noise_rate: float,
    ):

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
            if not i == nlayer - 1:
                self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer - 1:
                x = self.dropout_list[i](x)
        return x


class NicheTrans_ct(nn.Module):
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1):
        super(NicheTrans_ct, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate

        self.fea_size, self.img_size = 256, 256

        self.encoder_rna = NetBlock(
            nlayer=2,
            dim_list=[source_length, 512, self.fea_size],
            dropout_rate=self.dropout_rate,
            noise_rate=self.noise_rate,
        )

        self.projection_rna = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.local_graph_encoder = LocalGraphTransformerEncoder(
            dim=self.fea_size,
            depth=1,
            heads=4,
            dim_head=64,
            dropout=self.dropout_rate,
            ff_mult=2,
        )

        self.dropout = nn.Dropout(self.dropout_rate)

        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(
                    nn.Linear(self.fea_size, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 1, bias=False),
                )
            )
        self.predict_layers = nn.ModuleList(predict_net)

        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.cell_tokens = nn.Parameter(torch.randn((1, 1, 13, self.fea_size), requires_grad=True))

        trunc_normal_(self.cell_tokens, std=.02)
        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)

    def forward(self, source, source_neighbor, cell_inf):
        b = source.size(0)
        l = source_neighbor.size(1)

        node_inputs = torch.cat([source[:, None, :], source_neighbor], dim=1)
        valid_mask = infer_valid_node_mask(node_inputs)
        role_tokens = build_local_role_tokens(
            self.token_center,
            self.token_neigh_1,
            self.token_neigh_2,
            l,
            valid_mask=valid_mask,
        )
        class_tokens = (self.cell_tokens * cell_inf.unsqueeze(dim=-1)).sum(-2)
        class_tokens = class_tokens * valid_mask.unsqueeze(-1).to(class_tokens.dtype)

        f_omic = self.encoder_rna(node_inputs.reshape(-1, self.source_length)).reshape(b, -1, self.fea_size)
        f_omic = f_omic + role_tokens + class_tokens
        f_omic = self.projection_rna(f_omic)
        f_omic = self.local_graph_encoder(f_omic, valid_mask)

        f = self.dropout(f_omic[:, 0, :])

        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        return out
