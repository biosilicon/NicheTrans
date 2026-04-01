from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *


class NicheTrans_img(nn.Module):
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1):
        super(NicheTrans_img, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate

        self.fea_size, self.img_size = 256, 128

        resnet = torchvision.models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])

        self.img = nn.Sequential(
            nn.Linear(512, self.img_size),
            nn.BatchNorm1d(self.img_size),
            nn.LeakyReLU(),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder = NetBlock(
            nlayer=2,
            dim_list=[source_length, 512, self.fea_size],
            dropout_rate=self.dropout_rate,
            noise_rate=self.noise_rate,
        )
        self.local_graph_encoder = LocalGraphTransformerEncoder(
            dim=self.fea_size,
            depth=1,
            heads=4,
            dim_head=64,
            dropout=self.dropout_rate,
            ff_mult=2,
        )

        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(
                    nn.Linear(self.fea_size + self.img_size, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 1, bias=True),
                )
            )
        self.predict_layers = nn.ModuleList(predict_net)

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

    def forward(self, img, source, source_neighbor):
        b = img.size(0)
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

        f_omic = self.encoder(node_inputs.reshape(-1, self.source_length)).reshape(b, -1, self.fea_size)
        f_omic = f_omic + role_tokens
        f_omic = self.non_linear(f_omic)
        f_omic = self.local_graph_encoder(f_omic, valid_mask)

        f_img = self.pooling(self.base(img)).flatten(1)
        f_img = self.dropout_5(f_img)
        f_img = self.img(f_img)

        f = torch.cat([f_omic[:, 0, :], f_img], dim=1)
        f = self.dropout(f)

        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        return out
