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
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

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


class EdgeAwareGraphSelfAttention(nn.Module):
    def __init__(self, query_dim=512, edge_dim=11, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, heads),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(edge_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_mask, edge_attr):
        h = self.heads

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)

        edge_bias = rearrange(self.edge_bias(edge_attr), 'b i j h -> b h i j')
        edge_value = rearrange(self.edge_value(edge_attr), 'b i j (h d) -> b h i j d', h=h)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        sim = sim + edge_bias
        sim = sim.masked_fill(~adjacency_mask[:, None, :, :], -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        messages = v[:, :, None, :, :] + edge_value
        out = einsum('b h i j, b h i j d -> b h i d', attn, messages)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class LocalGraphTransformerLayer(nn.Module):
    def __init__(self, dim=256, edge_dim=11, heads=4, dim_head=64, dropout=0., ff_mult=2):
        super().__init__()
        self.attn = EdgeAwareGraphSelfAttention(
            query_dim=dim,
            edge_dim=edge_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.ffn = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_mask, edge_attr):
        attn_out, attn_weights = self.attn(self.norm1(x), adjacency_mask, edge_attr)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights


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
    ):
        super().__init__()
        self.edge_dim = 11
        self.neighbor_knn = neighbor_knn
        self.layers = nn.ModuleList(
            [
                LocalGraphTransformerLayer(
                    dim=dim,
                    edge_dim=self.edge_dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(depth)
            ]
        )

    def _build_neighbor_pair_mask(self, node_coords, valid_mask):
        num_neighbors = node_coords.size(1)
        neighbor_pair_mask = torch.zeros(
            (node_coords.size(0), num_neighbors, num_neighbors),
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
        eye = torch.eye(num_nodes, device=valid_mask.device, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1)
        adjacency_mask = eye.clone()

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

        valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        adjacency_mask = adjacency_mask & (valid_pairs | eye)
        return adjacency_mask

    def _derive_cell_type_ids(self, cell_inf, valid_mask):
        if cell_inf is None:
            return None, None

        cell_type_ids = cell_inf.argmax(dim=-1)
        cell_type_valid = cell_inf.abs().sum(dim=-1) > 0
        cell_type_valid = cell_type_valid & valid_mask
        return cell_type_ids, cell_type_valid

    def _build_edge_attr(self, node_coords, hop_ids, valid_mask, adjacency_mask, cell_inf=None):
        batch_size, num_nodes, _ = node_coords.shape
        dtype = node_coords.dtype

        rel_coords = node_coords[:, None, :, :] - node_coords[:, :, None, :]
        dx = rel_coords[..., 0]
        dy = rel_coords[..., 1]
        distance = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)

        center_distance = torch.sqrt(node_coords[:, :, 0].pow(2) + node_coords[:, :, 1].pow(2) + 1e-8)
        neighbor_valid_mask = valid_mask.clone()
        neighbor_valid_mask[:, 0] = False
        center_distance = center_distance.masked_fill(~neighbor_valid_mask, 0.0)
        neighbor_count = neighbor_valid_mask.sum(dim=-1).clamp(min=1).to(dtype)
        local_scale = center_distance.sum(dim=-1) / neighbor_count
        local_scale = torch.where(local_scale > 0, local_scale, torch.ones_like(local_scale))
        normalized_distance = distance / local_scale[:, None, None].clamp(min=1e-6)

        node_index = torch.arange(num_nodes, device=node_coords.device)
        query_is_center = (node_index[None, :, None] == 0).expand(batch_size, num_nodes, num_nodes)
        key_is_center = (node_index[None, None, :] == 0).expand(batch_size, num_nodes, num_nodes)
        is_self = torch.eye(num_nodes, device=node_coords.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
        is_neighbor_neighbor = ~(query_is_center | key_is_center | is_self)

        hop_i = hop_ids[:, :, None]
        hop_j = hop_ids[:, None, :]
        same_hop = hop_i == hop_j
        hop_delta = (hop_j - hop_i).abs().to(dtype)

        cell_type_ids, cell_type_valid = self._derive_cell_type_ids(cell_inf, valid_mask)
        if cell_type_ids is None:
            same_cell_type = torch.zeros((batch_size, num_nodes, num_nodes), device=node_coords.device, dtype=dtype)
        else:
            same_cell_type = ((cell_type_ids[:, :, None] == cell_type_ids[:, None, :]) & cell_type_valid[:, :, None] & cell_type_valid[:, None, :]).to(dtype)

        edge_attr = torch.stack(
            [
                dx,
                dy,
                distance,
                normalized_distance,
                query_is_center.to(dtype),
                key_is_center.to(dtype),
                is_self.to(dtype),
                is_neighbor_neighbor.to(dtype),
                same_hop.to(dtype),
                hop_delta,
                same_cell_type,
            ],
            dim=-1,
        )

        edge_attr = edge_attr * adjacency_mask.unsqueeze(-1).to(dtype)
        return edge_attr

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
        edge_attr = self._build_edge_attr(node_coords, hop_ids, valid_mask, adjacency_mask, cell_inf=cell_inf)

        return {
            'valid_mask': valid_mask,
            'hop_ids': hop_ids,
            'coords': node_coords,
            'adjacency_mask': adjacency_mask,
            'edge_attr': edge_attr,
        }

    def forward(self, x, valid_mask=None, graph_context=None, return_attention=False):
        if graph_context is None:
            if valid_mask is None:
                raise ValueError('Either valid_mask or graph_context must be provided.')
            graph_context = self.build_graph_context(x, graph_meta={'valid_mask': valid_mask})

        valid_mask = graph_context['valid_mask']
        adjacency_mask = graph_context['adjacency_mask']
        edge_attr = graph_context['edge_attr']
        node_mask = valid_mask.unsqueeze(-1).to(x.dtype)

        attention_maps = []
        for layer in self.layers:
            x = x * node_mask
            x, attn_weights = layer(x, adjacency_mask, edge_attr)
            x = x * node_mask
            if return_attention:
                attention_maps.append(attn_weights)

        if return_attention:
            graph_state = {
                'attention_weights': attention_maps,
                'adjacency_mask': adjacency_mask,
                'edge_attr': edge_attr,
                'hop_ids': graph_context['hop_ids'],
                'coords': graph_context['coords'],
                'valid_mask': valid_mask,
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
