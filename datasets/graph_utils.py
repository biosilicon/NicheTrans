from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def to_numpy_array(array_like) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if issparse(array_like):
        return array_like.toarray()
    return np.asarray(array_like)


def build_knn_edge_index(
    pos: np.ndarray,
    k: int = 6,
    force_undirected: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build graph connectivity and edge features from coordinates.

    Args:
        pos: Node coordinates with shape [N, 2].
        k: Number of nearest neighbors for each node.
        force_undirected: Whether to add the reverse edge for every kNN edge.

    Returns:
        edge_index: Graph connectivity with shape [2, E].
        edge_attr: Spatial edge features with shape [E, 3] = [dx, dy, dist].
    """
    num_nodes = int(pos.shape[0])
    if num_nodes <= 1:
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 3), dtype=torch.float32)
        return empty_index, empty_attr

    knn = min(k + 1, num_nodes)
    nbrs = NearestNeighbors(n_neighbors=knn).fit(pos)
    _, indices = nbrs.kneighbors(pos)

    edges = set()
    for src in range(num_nodes):
        for dst in indices[src, 1:]:
            dst = int(dst)
            if src == dst:
                continue
            edges.add((src, dst))
            if force_undirected:
                edges.add((dst, src))

    if not edges:
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 3), dtype=torch.float32)
        return empty_index, empty_attr

    ordered_edges = np.array(sorted(edges), dtype=np.int64)
    edge_index_np = ordered_edges.T

    src = edge_index_np[0]
    dst = edge_index_np[1]
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    dist = np.sqrt(dx**2 + dy**2)
    edge_attr_np = np.stack([dx, dy, dist], axis=1).astype(np.float32)

    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32)
    return edge_index, edge_attr


def build_node_masks(
    num_nodes: int,
    split: str,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    if split == "train":
        if num_nodes == 1 or val_ratio <= 0:
            train_mask[:] = True
            return train_mask, val_mask, test_mask

        rng = np.random.default_rng(seed)
        indices = np.arange(num_nodes)
        rng.shuffle(indices)

        num_val = int(round(num_nodes * val_ratio))
        num_val = min(max(num_val, 1), num_nodes - 1)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        return train_mask, val_mask, test_mask

    if split == "val":
        val_mask[:] = True
        return train_mask, val_mask, test_mask

    if split == "test":
        test_mask[:] = True
        return train_mask, val_mask, test_mask

    raise ValueError(f"Unsupported split '{split}'. Expected one of: train, val, test.")


def encode_cell_types(
    cell_types: Sequence,
    vocabulary: Sequence,
) -> torch.Tensor:
    vocab_to_index = {value: idx for idx, value in enumerate(vocabulary)}
    encoded = [vocab_to_index.get(value, -1) for value in cell_types]
    return torch.as_tensor(encoded, dtype=torch.long)


def build_slice_graph(
    node_features,
    node_targets,
    coordinates,
    split: str,
    *,
    k: int = 6,
    force_undirected: bool = True,
    val_ratio: float = 0.1,
    mask_seed: int = 0,
    cell_type: Optional[Sequence] = None,
    cell_type_vocabulary: Optional[Sequence] = None,
    node_ids: Optional[Iterable[str]] = None,
    sample_id: Optional[str] = None,
    slice_name: Optional[str] = None,
) -> Data:
    # x: [N, source_dim]
    x = torch.as_tensor(to_numpy_array(node_features), dtype=torch.float32)
    # y: [N, target_dim]
    y = torch.as_tensor(to_numpy_array(node_targets), dtype=torch.float32)
    # pos: [N, 2]
    pos_np = to_numpy_array(coordinates).astype(np.float32)
    pos = torch.as_tensor(pos_np, dtype=torch.float32)

    if x.dim() != 2:
        raise ValueError(f"Expected node features with shape [N, source_dim], got {tuple(x.shape)}.")
    if y.dim() != 2:
        raise ValueError(f"Expected node targets with shape [N, target_dim], got {tuple(y.shape)}.")
    if pos.shape != (x.size(0), 2):
        raise ValueError(
            "Coordinates must align one-to-one with nodes and have shape [N, 2]. "
            f"Got {tuple(pos.shape)} for {x.size(0)} nodes."
        )
    if y.size(0) != x.size(0):
        raise ValueError("Node targets must align one-to-one with node features.")

    edge_index, edge_attr = build_knn_edge_index(pos_np, k=k, force_undirected=force_undirected)
    train_mask, val_mask, test_mask = build_node_masks(
        num_nodes=x.size(0),
        split=split,
        val_ratio=val_ratio,
        seed=mask_seed,
    )

    data_kwargs = {
        "x": x,
        "y": y,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "node_index": torch.arange(x.size(0), dtype=torch.long),
    }

    if cell_type is not None and cell_type_vocabulary is not None:
        data_kwargs["cell_type"] = encode_cell_types(cell_type, cell_type_vocabulary)

    data = Data(**data_kwargs)
    data.slice_name = slice_name if slice_name is not None else sample_id
    data.sample_id = sample_id if sample_id is not None else slice_name
    if node_ids is not None:
        data.node_ids = list(node_ids)
    return data


def build_graph_dataloader(graphs, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


def count_graph_nodes(graphs) -> int:
    return int(sum(graph.num_nodes for graph in graphs))
