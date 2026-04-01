from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *
from model.nicheTrans import NetBlock as _SharedNetBlock


class NetBlock(_SharedNetBlock):
    """Compatibility wrapper around the shared encoder block."""


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
            depth=2,
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

    def forward(self, input, graph_meta=None, return_attention=False):
        b, n, _ = input.shape
        cell_inf = input[:, :, -13:]
        omics_data = input[:, :, :-13]
        l = n - 1

        graph_context = self.local_graph_encoder.build_graph_context(
            omics_data,
            graph_meta=graph_meta,
            cell_inf=cell_inf,
        )
        valid_mask = graph_context['valid_mask']
        role_tokens = build_local_role_tokens(
            self.token_center,
            self.token_neigh_1,
            self.token_neigh_2,
            l,
            valid_mask=valid_mask,
            hop_ids=graph_context['hop_ids'],
        )
        class_tokens = (self.cell_tokens * cell_inf.unsqueeze(dim=-1)).sum(-2)
        class_tokens = class_tokens * valid_mask.unsqueeze(-1).to(class_tokens.dtype)

        f_omic = self.encoder_rna(omics_data.reshape(-1, self.source_length)).reshape(b, n, self.fea_size)
        f_omic = f_omic + role_tokens + class_tokens
        f_omic = self.projection_rna(f_omic)
        if return_attention:
            f_omic, graph_state = self.local_graph_encoder(
                f_omic,
                graph_context=graph_context,
                return_attention=True,
            )
        else:
            f_omic = self.local_graph_encoder(f_omic, graph_context=graph_context)

        f = self.dropout(f_omic[:, 0, :])
        out = self.predict_layers[1](f)

        if return_attention:
            return out, graph_state

        return out
