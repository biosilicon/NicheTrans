from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *
from model.spot_type_utils import expand_spot_type_sequence, gather_token_bank


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
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1,
                 n_spot_types=1, n_cell_types=None):
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
            Vocabulary size of the global center-spot cell-type embedding.
        n_cell_types : int, optional
            Width of the soft cell-type composition vectors used in ``cell_inf``.
            Defaults to ``n_spot_types`` so a single unified global cell-type
            vocabulary is used throughout the model.
        """
        super(NicheTrans_ct, self).__init__()

        if n_cell_types is None:
            n_cell_types = n_spot_types

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.n_spot_types = n_spot_types
        self.n_cell_types = n_cell_types

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

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)

        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)

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

        # Each global cell type owns an independent pair of neighborhood tokens.
        self.token_neigh_1 = nn.Parameter(torch.randn((self.n_cell_types, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((self.n_cell_types, self.fea_size), requires_grad=True))
        self.token_center_emb = nn.Embedding(n_spot_types, self.fea_size)

        # ``cell_tokens`` now follows the same global cell-type vocabulary as
        # the data manager, instead of assuming a hard-coded size of 13.
        self.cell_tokens = nn.Parameter(torch.randn((1, 1, self.n_cell_types, self.fea_size), requires_grad=True))

        trunc_normal_(self.cell_tokens, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)
        trunc_normal_(self.token_center_emb.weight, std=.02)

    def _expand_legacy_neighborhood_token(self, token):
        if token.shape == (self.n_cell_types, self.fea_size):
            return token
        if token.ndim == 3 and token.shape == (1, 1, self.fea_size):
            token = token.view(1, self.fea_size)
        if token.ndim == 2 and token.shape == (1, self.fea_size):
            return token.expand(self.n_cell_types, -1).clone()
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

    def _get_token_spot_type_ids(self, batch_size, token_count, device, cell_inf=None, spot_type=None):
        if spot_type is not None:
            return expand_spot_type_sequence(
                spot_type=spot_type,
                batch_size=batch_size,
                token_count=token_count,
                device=device,
            )
        if cell_inf is not None:
            token_ids = cell_inf.argmax(dim=-1).long()
            token_ids[cell_inf.sum(dim=-1) <= 0] = -1
            return token_ids
        return torch.zeros((batch_size, token_count), dtype=torch.long, device=device)

    def _get_neighborhood_tokens(self, neighbor_spot_types):
        ring_length = neighbor_spot_types.size(1) // 2
        neigh1 = gather_token_bank(self.token_neigh_1, neighbor_spot_types[:, :ring_length])
        neigh2 = gather_token_bank(self.token_neigh_2, neighbor_spot_types[:, ring_length:])
        return neigh1, neigh2

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
            When provided the model adds a soft cell-type embedding on top of
            the spatial tokens.
        spot_type : LongTensor, shape (B,) or (B, 1+L), optional
            Integer global cell-type IDs. ``(B,)`` keeps the legacy
            center-broadcast behavior. ``(B, 1+L)`` provides one type ID per
            token in the same order as ``source_neighbor``. Negative IDs mark
            padded neighbors and produce zero spatial tokens.
        """
        b = source.size(0)
        l = source_neighbor.size(1)

        spot_type = self._get_token_spot_type_ids(
            batch_size=b,
            token_count=1 + l,
            device=source.device,
            cell_inf=cell_inf,
            spot_type=spot_type,
        )
        center_token = gather_token_bank(self.token_center_emb.weight, spot_type[:, :1])
        neigh1_tokens, neigh2_tokens = self._get_neighborhood_tokens(spot_type[:, 1:])

        spatial_tokens = torch.cat(
            [center_token,
             neigh1_tokens,
             neigh2_tokens],
            dim=1
        )

        if cell_inf is not None:
            classes_tokens = (self.cell_tokens * cell_inf.unsqueeze(dim=-1)).sum(-2)
        else:
            classes_tokens = 0

        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        f_omic = self.encoder_rna(omic_data).view(b, -1, self.fea_size)
        f_omic = f_omic + spatial_tokens + classes_tokens

        f_omic = self.projection_rna(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        f_omic = f_omic[:, 0, :]
        f = self.dropout(f_omic)

        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        return out
