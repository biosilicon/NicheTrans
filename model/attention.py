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
            nn.Dropout(dropout)
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


def build_local_role_tokens(token_center, token_neigh_1, token_neigh_2, num_neighbors, valid_mask=None):
    num_first_hop = num_neighbors // 2
    num_second_hop = num_neighbors - num_first_hop
    role_tokens = torch.cat(
        [
            token_center,
            token_neigh_1.repeat(1, num_first_hop, 1),
            token_neigh_2.repeat(1, num_second_hop, 1),
        ],
        dim=1,
    )

    if valid_mask is not None:
        role_tokens = role_tokens * valid_mask.unsqueeze(-1).to(role_tokens.dtype)

    return role_tokens


class GraphSelfAttention(nn.Module):
    def __init__(self, query_dim=512, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_mask):
        h = self.heads

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        sim = sim.masked_fill(~adjacency_mask[:, None, :, :], -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LocalGraphTransformerLayer(nn.Module):
    def __init__(self, dim=256, heads=4, dim_head=64, dropout=0., ff_mult=2):
        super().__init__()
        self.attn = GraphSelfAttention(query_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ffn = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, adjacency_mask):
        x = x + self.attn(self.norm1(x), adjacency_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class LocalGraphTransformerEncoder(nn.Module):
    def __init__(self, dim=256, depth=1, heads=4, dim_head=64, dropout=0., ff_mult=2, connect_neighbor_pairs=False):
        super().__init__()
        self.connect_neighbor_pairs = connect_neighbor_pairs
        self.layers = nn.ModuleList(
            [
                LocalGraphTransformerLayer(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(depth)
            ]
        )

    def build_adjacency_mask(self, valid_mask):
        batch_size, num_nodes = valid_mask.shape
        adjacency_mask = torch.eye(num_nodes, device=valid_mask.device, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1)

        if num_nodes > 1:
            neighbor_valid = valid_mask[:, 1:]
            adjacency_mask[:, 0, 1:] = neighbor_valid
            adjacency_mask[:, 1:, 0] = neighbor_valid

            if self.connect_neighbor_pairs:
                neighbor_mask = neighbor_valid[:, :, None] & neighbor_valid[:, None, :]
                adjacency_mask[:, 1:, 1:] = adjacency_mask[:, 1:, 1:] | neighbor_mask

        return adjacency_mask

    def forward(self, x, valid_mask):
        adjacency_mask = self.build_adjacency_mask(valid_mask)
        valid_mask = valid_mask.unsqueeze(-1).to(x.dtype)

        for layer in self.layers:
            x = x * valid_mask
            x = layer(x, adjacency_mask)

        return x * valid_mask


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
