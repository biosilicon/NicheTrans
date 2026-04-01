from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import NetBlock as _SharedNetBlock


class NetBlock(_SharedNetBlock):
    """Compatibility wrapper around the shared encoder block."""


class NicheTrans(nn.Module):
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate

        self.fea_size, self.img_size = 256, 128

        self.encoder = NetBlock(
            nlayer=2,
            dim_list=[source_length, 512, self.fea_size],
            dropout_rate=self.dropout_rate,
            noise_rate=self.noise_rate,
        )
        self.local_graph_encoder = LocalGraphTransformerEncoder(
            dim=self.fea_size,
            depth=2,
            heads=4,
            dim_head=64,
            dropout=self.dropout_rate,
            ff_mult=2,
        )

        self.predict_layers = nn.Sequential(
            nn.Linear(self.fea_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, target_length, bias=False),
        )

        self.non_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_5 = nn.Dropout(0.5)

        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))

        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)

    def forward(self, source, source_neighbor, graph_meta=None, return_attention=False):
        b = source.size(0)
        l = source_neighbor.size(1)

        node_inputs = torch.cat([source[:, None, :], source_neighbor], dim=1)
        graph_context = self.local_graph_encoder.build_graph_context(node_inputs, graph_meta=graph_meta)
        valid_mask = graph_context['valid_mask']
        role_tokens = build_local_role_tokens(
            self.token_center,
            self.token_neigh_1,
            self.token_neigh_2,
            l,
            valid_mask=valid_mask,
            hop_ids=graph_context['hop_ids'],
        )

        f_omic = self.encoder(node_inputs.reshape(-1, self.source_length)).reshape(b, -1, self.fea_size)
        f_omic = f_omic + role_tokens
        f_omic = self.non_linear(f_omic)
        if return_attention:
            f_omic, graph_state = self.local_graph_encoder(
                f_omic,
                graph_context=graph_context,
                return_attention=True,
            )
        else:
            f_omic = self.local_graph_encoder(f_omic, graph_context=graph_context)

        f = self.dropout(f_omic[:, 0, :])
        out = self.predict_layers(f)

        if return_attention:
            return out, graph_state

        return out
