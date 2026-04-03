from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *


class NicheTrans_img(StackedMoEModelMixin, nn.Module):
    def __init__(
        self,
        source_length=877,
        target_length=137,
        noise_rate=0.2,
        dropout_rate=0.1,
        use_moe_ffn=True,
        num_experts=1,
        moe_gate_hidden_dim=None,
        moe_gate_type='softmax',
        ffn_mult=2,
        moe_num_layers=1,
        moe_router_temperature_enable=False,
        moe_router_temperature_start=1.0,
        moe_router_temperature_mid=0.7,
        moe_router_temperature_end=0.5,
        moe_router_temperature_schedule="step",
        moe_balance_loss_enable=False,
        moe_balance_loss_weight=1e-3,
        moe_balance_loss_type="mse_uniform",
        moe_router_entropy_penalty_enable=False,
        moe_router_entropy_penalty_weight=1e-3,
    ):
        super(NicheTrans_img, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.use_moe_ffn = use_moe_ffn
        self.num_experts = max(int(num_experts), 1)
        self.moe_gate_hidden_dim = None if moe_gate_hidden_dim in (None, 0) else int(moe_gate_hidden_dim)
        self.moe_gate_type = moe_gate_type
        self.ffn_mult = int(ffn_mult)
        self.moe_num_layers = max(int(moe_num_layers), 1)

        self.fea_size, self.img_size = 256, 128

        ###########
        # image encoder
        resnet = torchvision.models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])

        self.img = nn.Sequential(nn.Linear(512, self.img_size),
                                 nn.BatchNorm1d(self.img_size),
                                 nn.LeakyReLU())

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        #############
        # omics encoder
        self.encoder = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        (
            self.fusion_omic,
            self.ffn_omic,
            self.ln1,
            self.ln2,
            self.extra_fusion_omic,
            self.extra_ffn_omic,
            self.extra_ln1,
            self.extra_ln2,
        ) = build_omic_block_stack(
            dim=self.fea_size,
            dropout=self.dropout_rate,
            mult=self.ffn_mult,
            num_layers=self.moe_num_layers,
            num_experts=self.num_experts,
            gate_hidden_dim=self.moe_gate_hidden_dim,
            use_moe=self.use_moe_ffn,
            gate_type=self.moe_gate_type,
            router_temperature_enable=moe_router_temperature_enable,
            router_temperature_start=moe_router_temperature_start,
            router_temperature_mid=moe_router_temperature_mid,
            router_temperature_end=moe_router_temperature_end,
            router_temperature_schedule=moe_router_temperature_schedule,
            balance_loss_enable=moe_balance_loss_enable,
            balance_loss_weight=moe_balance_loss_weight,
            balance_loss_type=moe_balance_loss_type,
            router_entropy_penalty_enable=moe_router_entropy_penalty_enable,
            router_entropy_penalty_weight=moe_router_entropy_penalty_weight,
        )

        ##############
        # prediction layers
        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(nn.Linear(self.fea_size + self.img_size, 128),
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

        ################
        # initialize tokens for semantic embedding
        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))

        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)
        ################

    def forward(self, img, source, source_neighbor, return_moe_info=False):
        b = img.size(0)

        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, 4, 1), self.token_neigh_2.repeat(1, 4, 1)], dim=1)

        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder(omic_data).view(b, -1, self.fea_size) 
        f_omic = f_omic + spatial_tokens

        f_omic = self.non_linear(f_omic)

        f_omic, routing_info = self.run_omic_blocks(f_omic, return_moe_info=return_moe_info)
        
        # image feature extraction
        f_img = self.pooling(self.base(img)).squeeze()
        f_img = self.dropout_5(f_img)
        f_img = self.img(f_img)

        f = torch.cat([f_omic[:, 0, :], f_img], dim=1) 
        f = self.dropout(f)

        # final prediction
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        if return_moe_info:
            return build_moe_output(out, routing_info, center_token_index=0)
        return out
