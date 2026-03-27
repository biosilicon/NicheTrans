"""
GNN-based niche/context encoder to replace the Transformer self-attention block.

Public API
----------
GNNNicheEncoder(in_dim, hidden_dim, num_layers, dropout, graph_type)
    forward(x: Tensor[b, n_nodes, in_dim]) -> Tensor[b, n_nodes, hidden_dim]

    Node 0 is always the center cell/spot (preserving the original convention).
    The caller extracts `out[:, 0, :]` for downstream prediction, exactly as in
    the Transformer path.

Graph topology
--------------
Two modes controlled by `graph_type`:
    "full"  – every node connected to every other node (mirrors Transformer
              all-to-all attention; 9-node niche → 72 directed edges)
    "star"  – center (node 0) connected bidirectionally to each neighbor only
              (hub-and-spoke; 9-node niche → 16 directed edges)

The edge_index template is built once per unique (n_nodes, graph_type) pair and
reused across batches.  Moving it to the right device costs nothing when already
resident there.

GNN architecture
----------------
    optional input projection (in_dim → hidden_dim, only when they differ)
    N × [SAGEConv → LayerNorm → LeakyReLU → Dropout  +  residual]
    All layers share hidden_dim so residual is always dimension-compatible
    after the (optional) first projection.

Dependencies
------------
Requires torch_geometric.  It is listed in requirements.txt for this project.
Install with:
    pip install torch-geometric
    pip install torch-scatter   (may also be needed depending on PyG version)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _build_single_edge_index(n_nodes: int, graph_type: str) -> torch.Tensor:
    """
    Return edge_index for ONE niche graph (no batch offset).

    Args:
        n_nodes:    1 (center) + k (neighbors)
        graph_type: "full" or "star"

    Returns:
        edge_index: LongTensor[2, E]
    """
    if graph_type == "star":
        # Bidirectional center ↔ neighbor edges only.
        # center→neighbor and neighbor→center for each of the k neighbors.
        src, dst = [], []
        for i in range(1, n_nodes):
            src.extend([0, i])
            dst.extend([i, 0])
    elif graph_type == "full":
        # All pairs excluding self-loops (directed).
        src = [i for i in range(n_nodes) for j in range(n_nodes) if i != j]
        dst = [j for i in range(n_nodes) for j in range(n_nodes) if i != j]
    else:
        raise ValueError(
            f"graph_type must be 'full' or 'star', got '{graph_type}'"
        )
    return torch.tensor([src, dst], dtype=torch.long)


def _batch_edge_index(
    edge_index_single: torch.Tensor,
    batch_size: int,
    n_nodes: int,
) -> torch.Tensor:
    """
    Tile a single-graph edge_index for a batch by offsetting node indices.

    Each sample i owns nodes [i*n_nodes, …, i*n_nodes + n_nodes - 1].

    Args:
        edge_index_single: LongTensor[2, E]   — single-graph template
        batch_size:        int                 — number of samples
        n_nodes:           int                 — nodes per graph

    Returns:
        edge_index: LongTensor[2, batch_size * E]
    """
    device = edge_index_single.device
    # offsets: [b, 1, 1] so it broadcasts over [1, 2, E]
    offsets = (torch.arange(batch_size, device=device) * n_nodes).view(-1, 1, 1)
    # edge_index_single: [2, E] → unsqueeze to [1, 2, E], add offsets → [b, 2, E]
    batched = edge_index_single.unsqueeze(0) + offsets
    return batched.view(2, -1)  # [2, b*E]


# ---------------------------------------------------------------------------
# Single GraphSAGE layer
# ---------------------------------------------------------------------------

class _SAGELayer(nn.Module):
    """
    One GraphSAGE message-passing step with post-conv norm, activation, dropout.

    SAGEConv: h_v = W · [h_v ‖ mean_{u∈N(v)} h_u]
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        assert _PYG_AVAILABLE, (
            "torch_geometric is required for GNNNicheEncoder. "
            "Install with: pip install torch-geometric"
        )
        # normalize=False because we apply LayerNorm immediately after
        self.conv = SAGEConv(in_dim, out_dim, normalize=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]  edge_index: [2, E]
        return self.drop(self.act(self.norm(self.conv(x, edge_index))))


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class GNNNicheEncoder(nn.Module):
    """
    Drop-in replacement for the Transformer self-attention + FFN block.

    Expects the same 3-D input tensor [b, n_nodes, in_dim] that the Transformer
    path receives after the omics encoder + spatial token addition.  Returns a
    tensor of the same shape [b, n_nodes, hidden_dim].

    Args:
        in_dim:      Feature dimension coming in (fea_size, typically 256).
        hidden_dim:  Hidden/output dim.  Defaults to in_dim so residuals are
                     always shape-compatible without an explicit projection.
        num_layers:  Number of GraphSAGE layers (2 or 3 recommended).
        dropout:     Dropout rate applied after each conv layer.
        graph_type:  "full" (all-to-all) or "star" (center-hub only).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        graph_type: str = "full",
    ):
        super().__init__()
        assert _PYG_AVAILABLE, (
            "torch_geometric is required for GNNNicheEncoder. "
            "Install with: pip install torch-geometric"
        )
        assert num_layers >= 1, "num_layers must be ≥ 1"
        assert graph_type in ("full", "star"), (
            f"graph_type must be 'full' or 'star', got '{graph_type}'"
        )

        hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.graph_type = graph_type

        # Optional input projection so that all GNN layers share hidden_dim
        # (ensures residual connections are always dimension-compatible).
        if in_dim != hidden_dim:
            self.input_proj: nn.Module = nn.Linear(in_dim, hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # All layers are hidden_dim → hidden_dim after the optional first proj.
        self.layers = nn.ModuleList([
            _SAGELayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Edge index template: built lazily on first forward call and reused.
        # Stored as a plain attribute (not a buffer) because its size changes
        # with n_nodes.  We move it to the target device in _get_edge_index.
        self._edge_tmpl: torch.Tensor | None = None
        self._edge_tmpl_n: int = -1          # n_nodes for which template was built

    # ------------------------------------------------------------------
    def _get_edge_index(
        self, n_nodes: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return batched edge_index, rebuilding the template only when needed."""
        if self._edge_tmpl is None or self._edge_tmpl_n != n_nodes:
            self._edge_tmpl = _build_single_edge_index(n_nodes, self.graph_type)
            self._edge_tmpl_n = n_nodes
        # .to() is a no-op when already on the right device
        return _batch_edge_index(self._edge_tmpl.to(device), batch_size, n_nodes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor[b, n_nodes, in_dim]
               n_nodes = 1 (center, index 0) + k (neighbors, indices 1..k)

        Returns:
            out: FloatTensor[b, n_nodes, hidden_dim]
                 out[:, 0, :] is the updated center-node embedding.
        """
        b, n_nodes, feat_dim = x.shape

        # Validate input dimensionality
        if feat_dim != self.in_dim:
            raise ValueError(
                f"GNNNicheEncoder.in_dim={self.in_dim} but got input "
                f"feature dim {feat_dim}. Check that fea_size matches."
            )
        if n_nodes < 2:
            raise ValueError(
                f"GNNNicheEncoder expects ≥ 2 nodes (center + ≥1 neighbor), "
                f"got {n_nodes}."
            )

        # Build batched edge_index: [2, b * E_per_graph]
        edge_index = self._get_edge_index(n_nodes, b, x.device)

        # Flatten across batch dimension: [b*n_nodes, in_dim]
        h = x.reshape(b * n_nodes, feat_dim)

        # Project to hidden_dim if necessary (Identity when dims match)
        h = self.input_proj(h)          # [b*n_nodes, hidden_dim]

        # GraphSAGE layers with residual connections.
        # All layers are hidden_dim → hidden_dim so residual is always valid.
        for layer in self.layers:
            h = layer(h, edge_index) + h    # [b*n_nodes, hidden_dim]

        # Restore batch dimension: [b, n_nodes, hidden_dim]
        return h.reshape(b, n_nodes, self.hidden_dim)
