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
        noise_rate: float
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
    

class NicheTrans_ct(StackedMoEModelMixin, nn.Module):
    # def __init__(self, rna_length=877, msi_length=137):
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
        super(NicheTrans_ct, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.use_moe_ffn = use_moe_ffn
        self.num_experts = num_experts
        self.moe_gate_hidden_dim = moe_gate_hidden_dim
        self.moe_gate_type = moe_gate_type
        self.ffn_mult = ffn_mult
        self.moe_num_layers = max(int(moe_num_layers), 1)

        self.fea_size, self.img_size = 256, 256

        self.encoder_rna = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        self.projection_rna = nn.Sequential(nn.Linear(256, 256),
                                            nn.LayerNorm(256),
                                            nn.ReLU(inplace=True))

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
        
        ################
        self.dropout = nn.Dropout(self.dropout_rate)

        # define the prediction layer
        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(nn.Linear(self.fea_size, 128),
                             nn.BatchNorm1d(128),
                             nn.LeakyReLU(),
                            #  nn.Linear(128, 1, bias=True))
                            nn.Linear(128, 1, bias=False))
            )
        self.predict_layers = nn.ModuleList(predict_net)

        ###############

        # to define and normalize the tokens
        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.cell_tokens = nn.Parameter(torch.randn((1, 1, 13, self.fea_size), requires_grad=True))

        trunc_normal_(self.cell_tokens, std=.02)
        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)
       
    def forward(self, input, return_moe_info=False):
        b = input.size(0)

        cell_inf = input[:, :, -13:]
        omics_data = input[:, :, 0: -13]

        classes_tokens = (self.cell_tokens * cell_inf.unsqueeze(dim=-1)).sum(-2)

        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, 6, 1), self.token_neigh_2.repeat(1, 6, 1)], dim=1)

        omic_data = omics_data.view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder_rna(omic_data).view(b, -1, self.fea_size) 
        f_omic = f_omic + spatial_tokens + classes_tokens

        f_omic = self.projection_rna(f_omic)

        f_omic, routing_info = self.run_omic_blocks(f_omic, return_moe_info=return_moe_info)

        f_omic = f_omic[:, 0, :]
        f = self.dropout(f_omic)

        # final prediction
        # out = []
        # for i in range(self.target_length):
        #     out.append(self.predict_layers[i](f))
        # out = torch.cat(out, dim=1)
        out = self.predict_layers[1](f)

        if return_moe_info:
            return build_moe_output(out, routing_info, center_token_index=0)
        return out
