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
    

class NicheTrans_ct(nn.Module):
    # def __init__(self, rna_length=877, msi_length=137):
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1,
                 n_spot_types=1):
        """
        Parameters
        ----------
        source_length : int
            Number of input omics features.
        target_length : int
            Number of prediction targets.
        noise_rate : float
            Dropout rate applied as input noise in the encoder.
        dropout_rate : float
            Dropout rate used in FC layers.
        n_spot_types : int
            Vocabulary size of the spot-type embedding.  Each cell type
            (derived from the existing ``ct_top`` annotation) gets its own
            independent learnable center spatial token.  Defaults to 1 for
            backward compatibility.
        """
        super(NicheTrans_ct, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.n_spot_types = n_spot_types

        self.fea_size, self.img_size = 256, 256

        self.encoder_rna = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        self.projection_rna = nn.Sequential(nn.Linear(256, 256),
                                            nn.LayerNorm(256),
                                            nn.ReLU(inplace=True))

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)

        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)

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

        # Spatial tokens for ring-level positional encoding.
        # token_neigh_1 / token_neigh_2 are shared across all spot types.
        # token_center_emb provides a per-spot-type learnable center token
        # indexed by the integer cell-type label (n_spot_types == number of
        # distinct ct_top labels in the training data).
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))

        # Per-spot-type center token: shape (n_spot_types, fea_size).
        # nn.Embedding is efficient, part of model.parameters() / state_dict(),
        # and makes adding new cell types trivial (just expand the embedding).
        self.token_center_emb = nn.Embedding(n_spot_types, self.fea_size)

        # cell_tokens encodes the soft, per-token cell-type composition via
        # one-hot weighted sum (existing NicheTrans_ct mechanism, retained for
        # the ct_information forward path).
        self.cell_tokens = nn.Parameter(torch.randn((1, 1, 13, self.fea_size), requires_grad=True))

        trunc_normal_(self.cell_tokens, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)
        trunc_normal_(self.token_center_emb.weight, std=.02)

    def forward(self, source, source_neighbor, cell_inf=None, spot_type=None):
        """
        Parameters
        ----------
        source : Tensor, shape (B, source_length)
            Center-spot RNA features.
        source_neighbor : Tensor, shape (B, L, source_length)
            Neighbor RNA features; L must be even (two equal rings).
        cell_inf : Tensor, shape (B, 1+L, n_cell_types), optional
            One-hot cell-type composition for center + every neighbor token.
            When provided the model adds a soft cell-type embedding
            (``cell_tokens * cell_inf``) on top of the spatial tokens.
        spot_type : LongTensor, shape (B,), optional
            Integer cell-type ID for the center spot.  Used to look up the
            per-type learnable center token via ``token_center_emb``.
            Falls back to type 0 when omitted.
        """
        b = source.size(0)
        l = source_neighbor.size(1)

        # ── Per-spot-type center token ────────────────────────────────────
        if spot_type is None:
            spot_type = torch.zeros(b, dtype=torch.long, device=source.device)
        # (B, fea_size) → (B, 1, fea_size)
        center_token = self.token_center_emb(spot_type).unsqueeze(1)

        spatial_tokens = torch.cat(
            [center_token,
             self.token_neigh_1.expand(b, l // 2, -1),
             self.token_neigh_2.expand(b, l // 2, -1)],
            dim=1
        )  # (B, 1+L, fea_size)

        # ── Optional soft cell-type bias (existing ct mechanism) ──────────
        if cell_inf is not None:
            classes_tokens = (self.cell_tokens * cell_inf.unsqueeze(dim=-1)).sum(-2)
        else:
            classes_tokens = 0

        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder_rna(omic_data).view(b, -1, self.fea_size)
        f_omic = f_omic + spatial_tokens + classes_tokens

        f_omic = self.projection_rna(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        f_omic = f_omic[:, 0, :]
        f = self.dropout(f_omic)

        # final prediction
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        return out