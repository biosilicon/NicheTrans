from __future__ import absolute_import

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class NodeFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        noise_rate: float,
    ):
        super().__init__()
        self.noise_dropout = nn.Dropout(noise_rate)
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, 512),
                nn.Linear(512, hidden_dim),
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.BatchNorm1d(512),
                nn.BatchNorm1d(hidden_dim),
            ]
        )
        self.activations = nn.ModuleList([nn.LeakyReLU(), nn.LeakyReLU()])
        self.dropout = nn.Dropout(dropout_rate)

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, source_dim]
        x = self.noise_dropout(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[idx](x)
            x = self.activations[idx](x)
            if idx < len(self.layers) - 1:
                x = self.dropout(x)
        # x: [N, hidden_dim]
        return x


class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        heads: int,
        dropout_rate: float,
        ffn_multiplier: int = 2,
    ):
        super().__init__()
        self.attn = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            beta=False,
            dropout=dropout_rate,
            edge_dim=edge_dim,
        )
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # x: [N, hidden_dim]
        attn_out = self.attn(x, edge_index, edge_attr)
        x = self.norm1(x + self.attn_dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.ffn_dropout(ffn_out))
        # x: [N, hidden_dim]
        return x


class WholeSliceGraphTransformer(nn.Module):
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = 3,
        dropout_rate: float = 0.1,
        noise_rate: float = 0.2,
        use_cell_type: bool = False,
        num_cell_types: int = 0,
    ):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.use_cell_type = use_cell_type and num_cell_types > 0

        self.node_encoder = NodeFeatureEncoder(
            input_dim=source_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            noise_rate=noise_rate,
        )

        if self.use_cell_type:
            self.cell_type_embedding = nn.Embedding(num_cell_types, hidden_dim)
        else:
            self.cell_type_embedding = None

        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    heads=heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, target_dim),
        )

    def _add_cell_type_embedding(self, x: torch.Tensor, cell_type: torch.Tensor) -> torch.Tensor:
        if self.cell_type_embedding is None or cell_type is None:
            return x

        valid_mask = cell_type >= 0
        if not torch.any(valid_mask):
            return x

        cell_type_feature = torch.zeros_like(x)
        cell_type_feature[valid_mask] = self.cell_type_embedding(cell_type[valid_mask])
        return x + cell_type_feature

    def forward(self, data) -> torch.Tensor:
        # data.x: [N, source_dim]
        # data.edge_index: [2, E]
        # data.edge_attr: [E, edge_dim]
        x = self.node_encoder(data.x)

        cell_type = getattr(data, "cell_type", None)
        x = self._add_cell_type_embedding(x, cell_type)

        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)

        # pred: [N, target_dim]
        pred = self.predictor(x)
        return pred
