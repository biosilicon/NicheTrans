import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            kwargs.update(context=self.norm_context(context))

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim=512, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Self_Attention(nn.Module):
    def __init__(self, query_dim=512, context_dim=512, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, mask=None):
        h = self.heads

        q = self.to_q(x1)
        k, v = self.to_k(x1), self.to_v(x1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


def infer_valid_node_mask(node_inputs):
    valid_mask = node_inputs.abs().sum(dim=-1) > 0
    valid_mask[:, 0] = True
    return valid_mask


def infer_hop_ids(num_nodes, batch_size, device):
    hop_ids = torch.ones((batch_size, num_nodes), device=device, dtype=torch.long)
    hop_ids[:, 0] = 0

    if num_nodes > 1:
        num_neighbors = num_nodes - 1
        num_first_hop = num_neighbors // 2
        hop_ids[:, 1 + num_first_hop:] = 2

    return hop_ids


def build_fallback_coords(hop_ids, valid_mask, dtype):
    batch_size, num_nodes = hop_ids.shape
    coords = torch.zeros((batch_size, num_nodes, 2), device=hop_ids.device, dtype=dtype)

    if num_nodes == 1:
        return coords

    for hop_value, radius in ((1, 1.0), (2, 2.0)):
        hop_mask = (hop_ids == hop_value) & valid_mask
        for batch_index in range(batch_size):
            node_indices = torch.nonzero(hop_mask[batch_index], as_tuple=False).flatten()
            count = int(node_indices.numel())
            if count == 0:
                continue
            angles = torch.linspace(
                0,
                2 * math.pi,
                steps=count + 1,
                device=hop_ids.device,
                dtype=dtype,
            )[:-1]
            coords[batch_index, node_indices, 0] = torch.cos(angles) * radius
            coords[batch_index, node_indices, 1] = torch.sin(angles) * radius

    return coords


def build_local_role_tokens(token_center, token_neigh_1, token_neigh_2, num_neighbors, valid_mask=None, hop_ids=None):
    if hop_ids is None:
        batch_size = 1 if valid_mask is None else valid_mask.size(0)
        device = token_center.device
        hop_ids = infer_hop_ids(num_neighbors + 1, batch_size, device)

    role_bank = torch.cat([token_center, token_neigh_1, token_neigh_2], dim=1)
    role_tokens = role_bank[:, hop_ids.clamp(min=0, max=2), :]

    if role_tokens.dim() == 4:
        role_tokens = role_tokens.squeeze(0)

    if valid_mask is not None:
        role_tokens = role_tokens * valid_mask.unsqueeze(-1).to(role_tokens.dtype)

    return role_tokens


def move_graph_meta_to_device(graph_meta, device):
    if graph_meta is None:
        return None

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in graph_meta.items()
    }


def apply_neighbor_mask_to_graph_meta(graph_meta, neighbor_keep_mask):
    if graph_meta is None:
        return None

    updated_graph_meta = {}
    for key, value in graph_meta.items():
        updated_graph_meta[key] = value.clone() if torch.is_tensor(value) else value

    keep_mask = neighbor_keep_mask.squeeze(-1).bool()
    if 'valid_mask' in updated_graph_meta:
        updated_graph_meta['valid_mask'][:, 1:] = updated_graph_meta['valid_mask'][:, 1:] & keep_mask

    return updated_graph_meta


class PairwiseStructureBuilder(nn.Module):
    ROLE_NAMES = ('center', 'first_hop', 'second_hop')
    EDGE_TYPE_NAMES = ('self', 'center_edge', 'same_hop_edge', 'cross_hop_edge', 'invalid')

    def __init__(
        self,
        num_roles=3,
        num_direction_bins=8,
        distance_bin_edges=(0.5, 0.9, 1.25, 1.75, 2.5, 3.5),
        max_shortest_path_distance=4,
        max_degree=16,
        max_cell_types=64,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.num_direction_bins = num_direction_bins
        self.max_shortest_path_distance = max_shortest_path_distance
        self.max_degree = max_degree
        self.max_cell_types = max_cell_types

        self.register_buffer('distance_bin_edges', torch.tensor(distance_bin_edges, dtype=torch.float32))

        self.continuous_dim = 6
        self.num_distance_buckets = len(distance_bin_edges) + 3
        self.distance_unknown_idx = self.num_distance_buckets - 1
        self.num_direction_buckets = num_direction_bins + 2
        self.direction_unknown_idx = self.num_direction_buckets - 1
        self.num_role_pair_buckets = (num_roles * num_roles) + 1
        self.role_pair_unknown_idx = self.num_role_pair_buckets - 1
        self.num_hop_delta_buckets = num_roles + 1
        self.hop_delta_unknown_idx = self.num_hop_delta_buckets - 1
        self.num_edge_types = len(self.EDGE_TYPE_NAMES)
        self.num_shortest_path_buckets = max_shortest_path_distance + 2
        self.shortest_path_unknown_idx = self.num_shortest_path_buckets - 1
        self.num_same_cell_type_buckets = 3

    def _derive_cell_type_ids(self, cell_inf, valid_mask):
        if cell_inf is None:
            zeros = torch.zeros_like(valid_mask, dtype=torch.long)
            return zeros, torch.zeros_like(valid_mask)

        cell_type_valid = (cell_inf.abs().sum(dim=-1) > 0) & valid_mask
        cell_type_ids = cell_inf.argmax(dim=-1).clamp(min=0, max=self.max_cell_types - 1) + 1
        cell_type_ids = torch.where(cell_type_valid, cell_type_ids, torch.zeros_like(cell_type_ids))
        return cell_type_ids, cell_type_valid

    def _build_local_scale(self, node_coords, valid_mask):
        dtype = node_coords.dtype
        neighbor_valid_mask = valid_mask.clone()
        neighbor_valid_mask[:, 0] = False

        center_distance = torch.sqrt(node_coords[:, :, 0].pow(2) + node_coords[:, :, 1].pow(2) + 1e-8)
        center_distance = center_distance.masked_fill(~neighbor_valid_mask, 0.0)
        neighbor_count = neighbor_valid_mask.sum(dim=-1).clamp(min=1).to(dtype)
        local_scale = center_distance.sum(dim=-1) / neighbor_count
        local_scale = torch.where(local_scale > 0, local_scale, torch.ones_like(local_scale))
        return local_scale.clamp(min=1e-6)

    def _build_shortest_path_bucket(self, adjacency_mask, valid_mask):
        batch_size, num_nodes, _ = adjacency_mask.shape
        device = adjacency_mask.device
        eye = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0)

        inf = float(self.shortest_path_unknown_idx)
        distance = torch.full((batch_size, num_nodes, num_nodes), inf, device=device)
        edge_mask = adjacency_mask & ~eye
        distance = distance.masked_fill(edge_mask, 1.0)
        distance = torch.where(eye, torch.zeros_like(distance), distance)

        for intermediate in range(num_nodes):
            via_intermediate = (
                distance[:, :, intermediate].unsqueeze(-1)
                + distance[:, intermediate, :].unsqueeze(-2)
            )
            distance = torch.minimum(distance, via_intermediate)

        shortest_path_bucket = distance.clamp(max=float(self.shortest_path_unknown_idx)).long()
        valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        shortest_path_bucket = torch.where(
            valid_pairs,
            shortest_path_bucket,
            torch.full_like(shortest_path_bucket, self.shortest_path_unknown_idx),
        )
        return shortest_path_bucket

    def forward(self, node_coords, hop_ids, valid_mask, adjacency_mask, cell_inf=None):
        batch_size, num_nodes, _ = node_coords.shape
        device = node_coords.device
        dtype = node_coords.dtype
        valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        pair_mask = adjacency_mask & valid_pairs
        eye = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0)

        rel_coords = node_coords[:, None, :, :] - node_coords[:, :, None, :]
        dx = rel_coords[..., 0]
        dy = rel_coords[..., 1]
        distance = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)
        distance = torch.where(eye, torch.zeros_like(distance), distance)

        local_scale = self._build_local_scale(node_coords, valid_mask)
        scaled = local_scale[:, None, None]
        normalized_dx = dx / scaled
        normalized_dy = dy / scaled
        normalized_distance = distance / scaled
        log_distance = torch.log1p(normalized_distance)

        continuous_features = torch.stack(
            [
                normalized_dx,
                normalized_dy,
                normalized_distance,
                log_distance,
                normalized_dx.abs(),
                normalized_dy.abs(),
            ],
            dim=-1,
        )
        continuous_features = continuous_features * pair_mask.unsqueeze(-1).to(dtype)

        distance_bucket = torch.bucketize(normalized_distance, self.distance_bin_edges).long() + 1
        distance_bucket = torch.where(eye, torch.zeros_like(distance_bucket), distance_bucket)
        distance_bucket = torch.where(
            valid_pairs,
            distance_bucket,
            torch.full_like(distance_bucket, self.distance_unknown_idx),
        )

        angle = torch.remainder(torch.atan2(dy, dx) + (2 * math.pi), 2 * math.pi)
        direction_bucket = torch.floor(
            angle / ((2 * math.pi) / float(self.num_direction_bins))
        ).long() + 1
        direction_bucket = direction_bucket.clamp(max=self.num_direction_bins)
        direction_bucket = torch.where(eye, torch.zeros_like(direction_bucket), direction_bucket)
        direction_bucket = torch.where(
            valid_pairs,
            direction_bucket,
            torch.full_like(direction_bucket, self.direction_unknown_idx),
        )

        role_ids = hop_ids.clamp(min=0, max=self.num_roles - 1)
        role_i = role_ids[:, :, None]
        role_j = role_ids[:, None, :]
        role_pair_id = (role_i * self.num_roles) + role_j
        role_pair_id = torch.where(
            valid_pairs,
            role_pair_id,
            torch.full_like(role_pair_id, self.role_pair_unknown_idx),
        )

        hop_delta = (role_j - role_i).abs()
        hop_delta = torch.where(
            valid_pairs,
            hop_delta,
            torch.full_like(hop_delta, self.hop_delta_unknown_idx),
        )

        query_is_center = role_i == 0
        key_is_center = role_j == 0
        center_edge = pair_mask & ~eye & (query_is_center | key_is_center)
        same_hop_edge = pair_mask & ~eye & ~center_edge & (role_i == role_j)
        cross_hop_edge = pair_mask & ~eye & ~center_edge & (role_i != role_j)

        edge_type = torch.full((batch_size, num_nodes, num_nodes), 4, device=device, dtype=torch.long)
        edge_type = torch.where(eye & valid_pairs, torch.zeros_like(edge_type), edge_type)
        edge_type = torch.where(center_edge, torch.ones_like(edge_type), edge_type)
        edge_type = torch.where(same_hop_edge, torch.full_like(edge_type, 2), edge_type)
        edge_type = torch.where(cross_hop_edge, torch.full_like(edge_type, 3), edge_type)

        shortest_path_bucket = self._build_shortest_path_bucket(adjacency_mask, valid_mask)

        degree = adjacency_mask.long().sum(dim=-1) - valid_mask.long()
        degree = degree.clamp(min=0, max=self.max_degree)
        degree_ids = torch.where(
            valid_mask,
            degree + 1,
            torch.zeros_like(degree),
        )

        cell_type_ids, cell_type_valid = self._derive_cell_type_ids(cell_inf, valid_mask)
        known_cell_pairs = cell_type_valid[:, :, None] & cell_type_valid[:, None, :]
        same_cell_type = torch.zeros((batch_size, num_nodes, num_nodes), device=device, dtype=torch.long)
        same_cell_type = torch.where(
            known_cell_pairs & (cell_type_ids[:, :, None] == cell_type_ids[:, None, :]),
            torch.ones_like(same_cell_type),
            same_cell_type,
        )
        same_cell_type = torch.where(
            known_cell_pairs & (cell_type_ids[:, :, None] != cell_type_ids[:, None, :]),
            torch.full_like(same_cell_type, 2),
            same_cell_type,
        )

        return {
            'pair_mask': pair_mask,
            'dx': dx,
            'dy': dy,
            'distance': distance,
            'normalized_distance': normalized_distance,
            'continuous_features': continuous_features,
            'distance_bucket': distance_bucket,
            'direction_bucket': direction_bucket,
            'role_ids': role_ids,
            'role_pair_id': role_pair_id,
            'hop_delta': hop_delta,
            'edge_type': edge_type,
            'shortest_path_bucket': shortest_path_bucket,
            'degree_ids': degree_ids,
            'cell_type_ids': cell_type_ids,
            'same_cell_type': same_cell_type,
        }


class PairwiseStructuralBias(nn.Module):
    def __init__(
        self,
        heads,
        hidden_dim,
        continuous_dim,
        num_distance_buckets,
        num_direction_buckets,
        num_role_pair_buckets,
        num_hop_delta_buckets,
        num_edge_types,
        num_shortest_path_buckets,
        max_degree,
        max_cell_types,
        num_same_cell_type_buckets=3,
    ):
        super().__init__()
        self.continuous_mlp = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.distance_embedding = nn.Embedding(num_distance_buckets, hidden_dim)
        self.direction_embedding = nn.Embedding(num_direction_buckets, hidden_dim)
        self.role_pair_embedding = nn.Embedding(num_role_pair_buckets, hidden_dim)
        self.hop_delta_embedding = nn.Embedding(num_hop_delta_buckets, hidden_dim)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.shortest_path_embedding = nn.Embedding(num_shortest_path_buckets, hidden_dim)
        self.degree_embedding = nn.Embedding(max_degree + 2, hidden_dim, padding_idx=0)
        self.cell_type_embedding = nn.Embedding(max_cell_types + 1, hidden_dim, padding_idx=0)
        self.same_cell_type_embedding = nn.Embedding(num_same_cell_type_buckets, hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, heads),
        )

    def forward(self, relation_features):
        pair_repr = self.continuous_mlp(relation_features['continuous_features'])
        pair_repr = pair_repr + self.distance_embedding(relation_features['distance_bucket'])
        pair_repr = pair_repr + self.direction_embedding(relation_features['direction_bucket'])
        pair_repr = pair_repr + self.role_pair_embedding(relation_features['role_pair_id'])
        pair_repr = pair_repr + self.hop_delta_embedding(relation_features['hop_delta'])
        pair_repr = pair_repr + self.edge_type_embedding(relation_features['edge_type'])
        pair_repr = pair_repr + self.shortest_path_embedding(relation_features['shortest_path_bucket'])
        pair_repr = pair_repr + self.same_cell_type_embedding(relation_features['same_cell_type'])

        degree_embed = self.degree_embedding(relation_features['degree_ids'])
        pair_repr = pair_repr + degree_embed[:, :, None, :] + degree_embed[:, None, :, :]

        cell_type_embed = self.cell_type_embedding(relation_features['cell_type_ids'])
        pair_repr = pair_repr + cell_type_embed[:, :, None, :] + cell_type_embed[:, None, :, :]

        bias = self.output(pair_repr)
        bias = bias * relation_features['pair_mask'].unsqueeze(-1).to(bias.dtype)
        return rearrange(bias, 'b i j h -> b h i j')


class RelationAwareGraphSelfAttention(nn.Module):
    def __init__(
        self,
        query_dim=512,
        heads=4,
        dim_head=64,
        dropout=0.,
        relation_hidden_dim=64,
        structure_builder=None,
    ):
        super().__init__()
        if structure_builder is None:
            raise ValueError('structure_builder must be provided for relation-aware attention.')

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

        self.structural_bias = PairwiseStructuralBias(
            heads=heads,
            hidden_dim=relation_hidden_dim,
            continuous_dim=structure_builder.continuous_dim,
            num_distance_buckets=structure_builder.num_distance_buckets,
            num_direction_buckets=structure_builder.num_direction_buckets,
            num_role_pair_buckets=structure_builder.num_role_pair_buckets,
            num_hop_delta_buckets=structure_builder.num_hop_delta_buckets,
            num_edge_types=structure_builder.num_edge_types,
            num_shortest_path_buckets=structure_builder.num_shortest_path_buckets,
            max_degree=structure_builder.max_degree,
            max_cell_types=structure_builder.max_cell_types,
            num_same_cell_type_buckets=structure_builder.num_same_cell_type_buckets,
        )

    def _apply_mask(self, logits, pair_mask, valid_mask):
        batch_size, _, num_nodes, _ = logits.shape
        eye = torch.eye(num_nodes, device=logits.device, dtype=torch.bool).unsqueeze(0)
        invalid_query_fallback = (~valid_mask)[:, :, None] & eye
        safe_pair_mask = pair_mask | invalid_query_fallback

        logits = logits.masked_fill(~safe_pair_mask[:, None, :, :], -torch.finfo(logits.dtype).max)
        attention = logits.softmax(dim=-1)
        attention = attention * pair_mask[:, None, :, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.dropout(attention)

    def forward(self, x, pair_mask, valid_mask, relation_features):
        h = self.heads

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)

        structural_bias = self.structural_bias(relation_features)
        logits = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        logits = logits + structural_bias
        attention = self._apply_mask(logits, pair_mask, valid_mask)

        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attention, structural_bias


class LocalGraphTransformerLayer(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=4,
        dim_head=64,
        dropout=0.,
        ff_mult=2,
        relation_hidden_dim=64,
        structure_builder=None,
    ):
        super().__init__()
        self.attn = RelationAwareGraphSelfAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            relation_hidden_dim=relation_hidden_dim,
            structure_builder=structure_builder,
        )
        self.ffn = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_mask, valid_mask, relation_features):
        attn_out, attn_weights, structural_bias = self.attn(
            self.norm1(x),
            pair_mask,
            valid_mask,
            relation_features,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights, structural_bias


class LocalGraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim=256,
        depth=2,
        heads=4,
        dim_head=64,
        dropout=0.,
        ff_mult=2,
        neighbor_knn=3,
        relation_hidden_dim=64,
        num_direction_bins=8,
        distance_bin_edges=(0.5, 0.9, 1.25, 1.75, 2.5, 3.5),
        max_shortest_path_distance=4,
        max_degree=16,
        max_cell_types=64,
    ):
        super().__init__()
        self.neighbor_knn = neighbor_knn
        self.structure_builder = PairwiseStructureBuilder(
            num_roles=3,
            num_direction_bins=num_direction_bins,
            distance_bin_edges=distance_bin_edges,
            max_shortest_path_distance=max_shortest_path_distance,
            max_degree=max_degree,
            max_cell_types=max_cell_types,
        )
        self.layers = nn.ModuleList(
            [
                LocalGraphTransformerLayer(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    relation_hidden_dim=relation_hidden_dim,
                    structure_builder=self.structure_builder,
                )
                for _ in range(depth)
            ]
        )

    def _build_neighbor_pair_mask(self, node_coords, valid_mask):
        batch_size, num_neighbors, _ = node_coords.shape
        neighbor_pair_mask = torch.zeros(
            (batch_size, num_neighbors, num_neighbors),
            device=node_coords.device,
            dtype=torch.bool,
        )

        if num_neighbors <= 1:
            return neighbor_pair_mask

        pairwise_distance = torch.cdist(node_coords, node_coords)
        invalid_pairs = ~(valid_mask[:, :, None] & valid_mask[:, None, :])
        pairwise_distance = pairwise_distance.masked_fill(invalid_pairs, float('inf'))
        diagonal = torch.eye(num_neighbors, device=node_coords.device, dtype=torch.bool).unsqueeze(0)
        pairwise_distance = pairwise_distance.masked_fill(diagonal, float('inf'))

        k = min(self.neighbor_knn, max(num_neighbors - 1, 1))
        if k == 0:
            return neighbor_pair_mask

        knn_distance, knn_indices = torch.topk(pairwise_distance, k=k, dim=-1, largest=False)
        finite_knn = torch.isfinite(knn_distance)
        neighbor_pair_mask.scatter_(-1, knn_indices, finite_knn)
        neighbor_pair_mask = neighbor_pair_mask | neighbor_pair_mask.transpose(1, 2)
        neighbor_pair_mask = neighbor_pair_mask & ~diagonal
        return neighbor_pair_mask

    def _build_fallback_neighbor_pair_mask(self, hop_ids, valid_mask):
        same_hop = hop_ids[:, :, None] == hop_ids[:, None, :]
        bridge_hop = (
            ((hop_ids[:, :, None] == 1) & (hop_ids[:, None, :] == 2))
            | ((hop_ids[:, :, None] == 2) & (hop_ids[:, None, :] == 1))
        )
        valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        diagonal = torch.eye(hop_ids.size(1), device=hop_ids.device, dtype=torch.bool).unsqueeze(0)
        return (same_hop | bridge_hop) & valid_pairs & ~diagonal

    def build_adjacency_mask(self, valid_mask, node_coords=None, hop_ids=None):
        batch_size, num_nodes = valid_mask.shape
        valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        eye = torch.eye(num_nodes, device=valid_mask.device, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1)
        adjacency_mask = eye & valid_pairs

        if num_nodes > 1:
            neighbor_valid = valid_mask[:, 1:]
            center_neighbor_edges = neighbor_valid[:, None, :]
            adjacency_mask[:, 0:1, 1:] = center_neighbor_edges
            adjacency_mask[:, 1:, 0:1] = center_neighbor_edges.transpose(1, 2)

            if exists(node_coords):
                neighbor_pair_mask = self._build_neighbor_pair_mask(node_coords[:, 1:, :], neighbor_valid)
            else:
                neighbor_pair_mask = self._build_fallback_neighbor_pair_mask(hop_ids[:, 1:], neighbor_valid)

            adjacency_mask[:, 1:, 1:] = adjacency_mask[:, 1:, 1:] | neighbor_pair_mask

        return adjacency_mask & valid_pairs

    def build_graph_context(self, node_inputs, graph_meta=None, cell_inf=None):
        batch_size, num_nodes, _ = node_inputs.shape
        device = node_inputs.device
        dtype = node_inputs.dtype

        if graph_meta is not None and 'valid_mask' in graph_meta:
            valid_mask = graph_meta['valid_mask'].to(device=device).bool()
        else:
            valid_mask = infer_valid_node_mask(node_inputs)
        valid_mask[:, 0] = True

        if graph_meta is not None and 'hop_ids' in graph_meta:
            hop_ids = graph_meta['hop_ids'].to(device=device).long()
        else:
            hop_ids = infer_hop_ids(num_nodes, batch_size, device)
        hop_ids[:, 0] = 0

        if graph_meta is not None and 'coords' in graph_meta:
            node_coords = graph_meta['coords'].to(device=device, dtype=dtype)
        else:
            node_coords = build_fallback_coords(hop_ids, valid_mask, dtype)

        adjacency_mask = self.build_adjacency_mask(valid_mask, node_coords=node_coords, hop_ids=hop_ids)
        relation_features = self.structure_builder(
            node_coords,
            hop_ids,
            valid_mask,
            adjacency_mask,
            cell_inf=cell_inf,
        )

        return {
            'valid_mask': valid_mask,
            'hop_ids': hop_ids,
            'coords': node_coords,
            'adjacency_mask': adjacency_mask,
            'edge_attr': relation_features['continuous_features'],
            'relation_features': relation_features,
        }

    def forward(self, x, valid_mask=None, graph_context=None, return_attention=False):
        if graph_context is None:
            if valid_mask is None:
                raise ValueError('Either valid_mask or graph_context must be provided.')
            graph_context = self.build_graph_context(x, graph_meta={'valid_mask': valid_mask})

        valid_mask = graph_context['valid_mask']
        adjacency_mask = graph_context['adjacency_mask']
        relation_features = graph_context['relation_features']
        node_mask = valid_mask.unsqueeze(-1).to(x.dtype)

        attention_maps = []
        structural_biases = []
        for layer in self.layers:
            x = x * node_mask
            x, attn_weights, structural_bias = layer(x, adjacency_mask, valid_mask, relation_features)
            x = x * node_mask
            if return_attention:
                attention_maps.append(attn_weights)
                structural_biases.append(structural_bias)

        if return_attention:
            graph_state = {
                'attention_weights': attention_maps,
                'structural_bias': structural_biases,
                'center_attention_weights': [layer_attn[:, :, 0, :] for layer_attn in attention_maps],
                'center_structural_bias': [layer_bias[:, :, 0, :] for layer_bias in structural_biases],
                'adjacency_mask': adjacency_mask,
                'edge_attr': relation_features['continuous_features'],
                'hop_ids': graph_context['hop_ids'],
                'coords': graph_context['coords'],
                'valid_mask': valid_mask,
                'relation_features': relation_features,
                'role_names': self.structure_builder.ROLE_NAMES,
                'edge_type_names': self.structure_builder.EDGE_TYPE_NAMES,
                'distance_bin_edges': self.structure_builder.distance_bin_edges.detach().cpu().tolist(),
                'num_direction_bins': self.structure_builder.num_direction_bins,
            }
            return x * node_mask, graph_state

        return x * node_mask


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
