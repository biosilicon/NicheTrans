import math
import warnings
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


class FeedForwardExpert(nn.Sequential):
    def __init__(self, dim=512, mult=4, dropout=0.):
        super().__init__(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )


class SoftmaxGate(nn.Module):
    def __init__(self, dim, num_experts, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(dim, num_experts)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts)
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).softmax(dim=-1)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim=512,
        mult=4,
        dropout=0.,
        num_experts=1,
        gate_hidden_dim=None,
        use_moe=True,
        gate_type='softmax'
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.dropout = dropout
        self.gate_type = gate_type
        self.gate_hidden_dim = gate_hidden_dim
        self.num_experts = max(int(num_experts), 1) if use_moe else 1
        self.use_moe = self.num_experts > 1

        # Keep the first expert under the legacy `net` name so old checkpoints
        # continue to map cleanly when num_experts == 1.
        self.net = FeedForwardExpert(dim=dim, mult=mult, dropout=dropout)
        self.extra_experts = nn.ModuleList([
            FeedForwardExpert(dim=dim, mult=mult, dropout=dropout)
            for _ in range(self.num_experts - 1)
        ])

        if self.use_moe:
            if self.gate_type != 'softmax':
                raise ValueError(f"Unsupported gate_type: {self.gate_type}")
            self.gate = SoftmaxGate(
                dim=dim,
                num_experts=self.num_experts,
                hidden_dim=gate_hidden_dim
            )
        else:
            self.gate = None

    def forward(self, x):
        if not self.use_moe:
            return self.net(x)

        expert_outputs = [self.net(x)]
        expert_outputs.extend(expert(x) for expert in self.extra_experts)
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        mixture_weights = self.gate(x).unsqueeze(dim=-1)
        return (expert_outputs * mixture_weights).sum(dim=-2)

    def _prepare_legacy_moe_state_dict(self, state_dict, prefix):
        if self.use_moe:
            legacy_prefix = f"{prefix}net."
            legacy_keys = [key for key in state_dict.keys() if key.startswith(legacy_prefix)]
            if legacy_keys:
                warnings.warn(
                    "Loading a legacy single-expert FFN checkpoint into an MoE FFN. "
                    "The original FFN weights are copied into every expert and the gate "
                    "keeps its current initialization.",
                    RuntimeWarning
                )
                for expert_idx in range(len(self.extra_experts)):
                    expert_prefix = f"{prefix}extra_experts.{expert_idx}."
                    for legacy_key in legacy_keys:
                        suffix = legacy_key[len(legacy_prefix):]
                        expert_key = expert_prefix + suffix
                        if expert_key not in state_dict:
                            state_dict[expert_key] = state_dict[legacy_key].clone()

                for gate_key, gate_value in self.gate.state_dict().items():
                    full_gate_key = f"{prefix}gate.{gate_key}"
                    if full_gate_key not in state_dict:
                        state_dict[full_gate_key] = gate_value.detach().clone()
        else:
            moe_only_prefixes = (f"{prefix}extra_experts.", f"{prefix}gate.")
            removable_keys = [
                key for key in state_dict.keys()
                if key.startswith(moe_only_prefixes)
            ]
            for key in removable_keys:
                state_dict.pop(key)
            if removable_keys:
                warnings.warn(
                    "Loading an MoE FFN checkpoint into a single-expert FFN. "
                    "Additional expert and gate weights are ignored.",
                    RuntimeWarning
                )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs
    ):
        self._prepare_legacy_moe_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
        )


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
