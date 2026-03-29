from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *
from model.spot_type_utils import expand_spot_type_sequence, gather_token_bank


class NicheTrans_img(nn.Module):
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
            Vocabulary size of the spot-type embedding.  Each spot type gets
            its own independent learnable center spatial token.  Defaults to 1
            for backward compatibility (single shared token).
        """
        super(NicheTrans_img, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.n_spot_types = n_spot_types

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

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)

        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)

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
        # Spatial tokens for ring-level positional encoding.
        # token_neigh_1 / token_neigh_2 are now indexed from a per-spot-type
        # token bank. token_center_emb remains a per-spot-type center token.
        self.token_neigh_1 = nn.Parameter(torch.randn((self.n_spot_types, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((self.n_spot_types, self.fea_size), requires_grad=True))

        # Per-spot-type center token embedding: shape (n_spot_types, fea_size).
        self.token_center_emb = nn.Embedding(n_spot_types, self.fea_size)

        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)
        trunc_normal_(self.token_center_emb.weight, std=.02)

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

    def _get_neighborhood_tokens(self, neighbor_spot_types):
        ring_length = neighbor_spot_types.size(1) // 2
        neigh1 = gather_token_bank(self.token_neigh_1, neighbor_spot_types[:, :ring_length])
        neigh2 = gather_token_bank(self.token_neigh_2, neighbor_spot_types[:, ring_length:])
        return neigh1, neigh2

    def forward(self, img, source, source_neighbor, spot_type=None):
        """
        Parameters
        ----------
        img : Tensor, shape (B, C, H, W)
            Histology patch for each center spot.
        source : Tensor, shape (B, source_length)
            Center-spot omics features.
        source_neighbor : Tensor, shape (B, 8, source_length)
            Neighbor omics features (4 inner + 4 outer ring).
        spot_type : LongTensor, shape (B,) or (B, 9), optional
            Integer spot-type IDs in ``[0, n_spot_types)``. ``(B,)`` keeps the
            legacy center-broadcast behavior. ``(B, 9)`` should follow
            ``[center, neigh1_x4, neigh2_x4]`` order. Negative IDs mark
            padded neighbors and produce zero spatial tokens.
        """
        b = img.size(0)

        # ── Spot-type-specific center token ──────────────────────────────
        spot_type = expand_spot_type_sequence(
            spot_type=spot_type,
            batch_size=b,
            token_count=9,
            device=source.device,
        )
        center_token = gather_token_bank(self.token_center_emb.weight, spot_type[:, :1])
        neigh1_tokens, neigh2_tokens = self._get_neighborhood_tokens(spot_type[:, 1:])

        spatial_tokens = torch.cat(
            [center_token,
             neigh1_tokens,
             neigh2_tokens],
            dim=1
        )  # (B, 9, fea_size)

        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder(omic_data).view(b, -1, self.fea_size)
        f_omic = f_omic + spatial_tokens

        f_omic = self.non_linear(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

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

        return out
