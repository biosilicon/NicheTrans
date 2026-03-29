from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.spot_type_utils import expand_spot_type_sequence, gather_token_bank


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
        """
        Parameters
        ----------
        source_length : int
            Number of input omics features (e.g. HVGs).
        target_length : int
            Number of prediction targets (e.g. metabolites).
        noise_rate : float
            Dropout rate applied as input noise in the encoder.
        dropout_rate : float
            Dropout rate used in FC layers.
        n_spot_types : int
            Vocabulary size of the spot-type embedding.  Each spot type gets
            its own independent learnable ``center_spatial_token``.  Set to 1
            to recover the original single-token behaviour (backward-compatible
            default).
        """
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
        # Spatial tokens for ring-level positional encoding.
        # token_neigh_1 / token_neigh_2 are now spot-type-specific and are
        # indexed at runtime with the sample's spot type.
        # token_center_emb remains spot-type-specific as well.
        self.token_neigh_1 = nn.Parameter(torch.randn((self.n_spot_types, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((self.n_spot_types, self.fea_size), requires_grad=True))

        # Per-spot-type center token: shape (n_spot_types, fea_size).
        # nn.Embedding provides efficient integer-indexed lookup and is
        # included in model.parameters() / state_dict() automatically.
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


    def forward(self, source, source_neighbor, spot_type=None):
        """
        Parameters
        ----------
        source : Tensor, shape (B, source_length)
            Center-spot omics features.
        source_neighbor : Tensor, shape (B, L, source_length)
            Neighbor omics features; L must be even (split equally into two
            rings).
        spot_type : LongTensor, shape (B,) or (B, 1+L), optional
            Integer spot-type IDs in ``[0, n_spot_types)``. ``(B,)`` keeps the
            legacy behavior and broadcasts the center type to every token.
            ``(B, 1+L)`` provides one type ID per token in
            ``[center, first-order..., second-order...]`` order. Negative IDs
            mark padded neighbors and produce zero spatial tokens.
        """
        # 当前 batch 大小，即中心细胞/spot 的数量。
        b = source.size(0)
        # 邻居 token 数量，通常对应每个中心位置拼接进来的邻域样本数。
        l = source_neighbor.size(1)

        # ── Spot-type-specific center token ──────────────────────────────
        spot_type = expand_spot_type_sequence(
            spot_type=spot_type,
            batch_size=b,
            token_count=1 + l,
            device=source.device,
        )
        center_token = gather_token_bank(self.token_center_emb.weight, spot_type[:, :1])
        neigh1_tokens, neigh2_tokens = self._get_neighborhood_tokens(spot_type[:, 1:])

        # Assemble full spatial-token sequence: (B, 1+L, fea_size).
        # center_token is (B, 1, fea_size); ring tokens broadcast from (1, *, fea_size).
        spatial_tokens = torch.cat(
            [center_token,
             neigh1_tokens,
             neigh2_tokens],
            dim=1
        )  # (B, 1+L, fea_size)

        # 给中心样本增加一个 token 维度，形状从 [b, source_length] 变为 [b, 1, source_length]。
        source = source[:, None, :]
        # 将中心样本与邻居样本在 token 维拼接，再展平成二维张量，变成 [(b*(1+l)), source_length]
        # 以便共享同一个 encoder 对每个 token 的组学特征做编码。
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # 提取每个 token 的组学特征，再恢复成 [b, token_num, fea_size]。
        f_omic = self.encoder(omic_data).view(b, -1, self.fea_size)
        # 将可学习的空间 token 加到组学特征上，把空间角色信息注入表示中。
        # center_token 已经是 spot-type-specific，因此不同类型的 spot 会
        # 获得不同的空间偏置，ring tokens 仍共享。
        f_omic = f_omic + spatial_tokens

        # 经过一层非线性映射，进一步融合和变换特征。
        f_omic = self.non_linear(f_omic)

        # 自注意力残差块：先 LayerNorm，再做 token 间信息交互，最后与输入残差相加。
        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        # 前馈网络残差块：对每个 token 的表示逐位置变换，并保留残差连接。
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        # 仅取第 0 个 token（中心位置）的表示做最终预测，并施加 dropout。
        f = self.dropout(f_omic[:, 0, :])

        # 为每个目标基因/输出维度使用一个独立的预测头。
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        # 将所有单维预测结果拼接成最终输出 [b, target_length]。
        out = torch.cat(out, dim=1)

        return out
