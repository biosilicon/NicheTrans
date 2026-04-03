import math
import warnings
from collections.abc import Mapping
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
    def __init__(
        self,
        dim,
        num_experts,
        hidden_dim=None,
        temperature_enable=False,
        temperature_start=1.0,
        temperature_mid=0.7,
        temperature_end=0.5,
        temperature_schedule="step",
    ):
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

        self.temperature_enable = bool(temperature_enable)
        self.temperature_start = float(temperature_start)
        self.temperature_mid = float(temperature_mid)
        self.temperature_end = float(temperature_end)
        self.temperature_schedule = str(temperature_schedule).lower()
        self.current_epoch = 1

    def set_current_epoch(self, epoch):
        self.current_epoch = max(int(epoch), 1)

    def get_router_temperature(self):
        if not self.temperature_enable:
            return 1.0

        epoch = max(int(self.current_epoch), 1)
        if self.temperature_schedule == "step":
            if epoch <= 5:
                return self.temperature_start
            if epoch <= 10:
                return self.temperature_mid
            return self.temperature_end

        if self.temperature_schedule == "linear":
            if epoch <= 5:
                progress = 0.0 if epoch <= 1 else float(epoch - 1) / 4.0
                return self.temperature_start + progress * (self.temperature_mid - self.temperature_start)
            if epoch <= 10:
                progress = float(epoch - 6) / 4.0
                return self.temperature_mid + progress * (self.temperature_end - self.temperature_mid)
            return self.temperature_end

        raise ValueError(f"Unsupported router temperature schedule: {self.temperature_schedule}")

    def forward(self, x):
        gate_logits = self.net(x)
        tau = max(float(self.get_router_temperature()), 1e-6)
        gate_weights = (gate_logits / tau).softmax(dim=-1)
        return gate_weights, gate_logits, tau


def compute_expert_entropy(weights, eps=1e-12):
    clipped = weights.clamp_min(eps)
    return -(clipped * clipped.log()).sum(dim=-1)


def compute_gate_margin(weights):
    num_experts = weights.shape[-1]
    if num_experts <= 1:
        return torch.zeros(*weights.shape[:-1], device=weights.device, dtype=weights.dtype)
    if num_experts == 2:
        return (weights[..., 0] - weights[..., 1]).abs()
    top2 = weights.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def compute_expert_output_similarity(expert_outputs, eps=1e-8):
    if expert_outputs.shape[-2] <= 1:
        zero = expert_outputs.new_tensor(0.0)
        return zero, zero

    if expert_outputs.ndim == 4:
        features = expert_outputs.mean(dim=1)
    elif expert_outputs.ndim == 3:
        features = expert_outputs
    else:
        raise ValueError(
            "Expert output similarity expects expert outputs with shape [B, E, D] or [B, L, E, D]."
        )

    num_experts = features.shape[1]
    pair_index = torch.triu_indices(num_experts, num_experts, offset=1, device=features.device)
    left = features[:, pair_index[0], :]
    right = features[:, pair_index[1], :]
    cosine_values = F.cosine_similarity(left, right, dim=-1, eps=eps)
    return cosine_values.mean(), cosine_values.std(unbiased=False)


def build_moe_output(predictions, routing_info, center_token_index=0):
    """Package prediction tensors together with center-token MoE routing metadata."""
    if routing_info is None:
        return {"predictions": predictions, "moe_info": None}

    gate_weights = routing_info["gate_weights"]
    if gate_weights.ndim >= 3:
        center_gate_weights = gate_weights[:, center_token_index, :]
    else:
        center_gate_weights = gate_weights

    moe_info = {
        **routing_info,
        "center_token_index": center_token_index,
        "center_gate_weights": center_gate_weights,
        "center_top1_expert": center_gate_weights.argmax(dim=-1),
        "center_entropy": compute_expert_entropy(center_gate_weights),
        "center_gate_margin": compute_gate_margin(center_gate_weights),
    }
    return {"predictions": predictions, "moe_info": moe_info}


def _routing_metric_as_tensor(routing_info, key):
    value = routing_info.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value

    reference = routing_info.get("router_temperature")
    if not torch.is_tensor(reference):
        reference = routing_info.get("gate_weights")

    if torch.is_tensor(reference):
        return reference.new_tensor(float(value))
    return torch.tensor(float(value))


def aggregate_moe_routing_info(routing_infos):
    valid_infos = [info for info in routing_infos if isinstance(info, Mapping)]
    if not valid_infos:
        return None

    primary_info = dict(valid_infos[-1])
    primary_info["moe_num_layers"] = len(valid_infos)
    primary_info["layer_routing_info"] = list(valid_infos)

    sum_keys = ("moe_aux_loss",)
    mean_keys = (
        "router_temperature",
        "balance_loss",
        "router_entropy_penalty",
        "mean_gate_margin",
        "std_gate_margin",
        "expert_output_cosine_mean",
        "expert_output_cosine_std",
    )

    for key in sum_keys:
        tensors = [
            value
            for value in (_routing_metric_as_tensor(info, key) for info in valid_infos)
            if value is not None
        ]
        if tensors:
            primary_info[key] = torch.stack(tensors, dim=0).sum(dim=0)

    for key in mean_keys:
        tensors = [
            value
            for value in (_routing_metric_as_tensor(info, key) for info in valid_infos)
            if value is not None
        ]
        if tensors:
            primary_info[key] = torch.stack(tensors, dim=0).mean(dim=0)

    return primary_info


def build_omic_block_stack(
    dim,
    dropout,
    mult,
    num_layers,
    num_experts,
    gate_hidden_dim,
    use_moe,
    gate_type,
    router_temperature_enable,
    router_temperature_start,
    router_temperature_mid,
    router_temperature_end,
    router_temperature_schedule,
    balance_loss_enable,
    balance_loss_weight,
    balance_loss_type,
    router_entropy_penalty_enable,
    router_entropy_penalty_weight,
    heads=4,
    dim_head=64,
):
    def build_single_block():
        return (
            Self_Attention(
                query_dim=dim,
                context_dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
            FeedForward(
                dim=dim,
                mult=mult,
                dropout=dropout,
                num_experts=num_experts,
                gate_hidden_dim=gate_hidden_dim,
                use_moe=use_moe,
                gate_type=gate_type,
                router_temperature_enable=router_temperature_enable,
                router_temperature_start=router_temperature_start,
                router_temperature_mid=router_temperature_mid,
                router_temperature_end=router_temperature_end,
                router_temperature_schedule=router_temperature_schedule,
                balance_loss_enable=balance_loss_enable,
                balance_loss_weight=balance_loss_weight,
                balance_loss_type=balance_loss_type,
                router_entropy_penalty_enable=router_entropy_penalty_enable,
                router_entropy_penalty_weight=router_entropy_penalty_weight,
            ),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
        )

    total_layers = max(int(num_layers), 1)
    fusion_omic, ffn_omic, ln1, ln2 = build_single_block()
    extra_fusion_omic = nn.ModuleList()
    extra_ffn_omic = nn.ModuleList()
    extra_ln1 = nn.ModuleList()
    extra_ln2 = nn.ModuleList()

    for _ in range(total_layers - 1):
        block_fusion, block_ffn, block_ln1, block_ln2 = build_single_block()
        extra_fusion_omic.append(block_fusion)
        extra_ffn_omic.append(block_ffn)
        extra_ln1.append(block_ln1)
        extra_ln2.append(block_ln2)

    return (
        fusion_omic,
        ffn_omic,
        ln1,
        ln2,
        extra_fusion_omic,
        extra_ffn_omic,
        extra_ln1,
        extra_ln2,
    )


class StackedMoEModelMixin:
    _OMIC_EXTRA_MODULE_NAMES = (
        "extra_fusion_omic",
        "extra_ffn_omic",
        "extra_ln1",
        "extra_ln2",
    )

    def iter_omic_block_modules(self):
        yield self.fusion_omic, self.ffn_omic, self.ln1, self.ln2

        extra_fusion_omic = getattr(self, "extra_fusion_omic", [])
        extra_ffn_omic = getattr(self, "extra_ffn_omic", [])
        extra_ln1 = getattr(self, "extra_ln1", [])
        extra_ln2 = getattr(self, "extra_ln2", [])

        for block_index in range(len(extra_ffn_omic)):
            yield (
                extra_fusion_omic[block_index],
                extra_ffn_omic[block_index],
                extra_ln1[block_index],
                extra_ln2[block_index],
            )

    def run_omic_blocks(self, f_omic, return_moe_info=False):
        routing_infos = [] if return_moe_info else None

        for fusion_omic, ffn_omic, ln1, ln2 in self.iter_omic_block_modules():
            f_omic = fusion_omic(ln1(f_omic)) + f_omic
            if return_moe_info:
                ffn_out, routing_info = ffn_omic(ln2(f_omic), return_routing=True)
                routing_infos.append(routing_info)
            else:
                ffn_out = ffn_omic(ln2(f_omic))
            f_omic = ffn_out + f_omic

        routing_info = aggregate_moe_routing_info(routing_infos) if return_moe_info else None
        return f_omic, routing_info

    def set_current_epoch(self, epoch):
        self.current_epoch = max(int(epoch), 1)
        for _, ffn_omic, _, _ in self.iter_omic_block_modules():
            if hasattr(ffn_omic, "set_current_epoch"):
                ffn_omic.set_current_epoch(self.current_epoch)

    def _prepare_legacy_omic_stack_state_dict(self, state_dict, prefix):
        extra_prefixes = tuple(f"{prefix}{name}." for name in self._OMIC_EXTRA_MODULE_NAMES)
        has_extra_blocks = len(getattr(self, "extra_ffn_omic", [])) > 0

        if not has_extra_blocks:
            removable_keys = [key for key in state_dict.keys() if key.startswith(extra_prefixes)]
            for key in removable_keys:
                state_dict.pop(key)
            if removable_keys:
                warnings.warn(
                    "Loading a stacked omic-block checkpoint into a single-block model. "
                    "Additional omic blocks are ignored.",
                    RuntimeWarning
                )
            return

        missing_extra_keys = []
        for module_name in self._OMIC_EXTRA_MODULE_NAMES:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for key, value in module.state_dict().items():
                full_key = f"{prefix}{module_name}.{key}"
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach().clone()
                    missing_extra_keys.append(full_key)

        if missing_extra_keys:
            warnings.warn(
                "Loading a single-block checkpoint into a stacked omic-block model. "
                "Additional omic blocks keep their current initialization.",
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
        self._prepare_legacy_omic_stack_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim=512,
        mult=4,
        dropout=0.,
        num_experts=1,
        gate_hidden_dim=None,
        use_moe=True,
        gate_type='softmax',
        router_temperature_enable=False,
        router_temperature_start=1.0,
        router_temperature_mid=0.7,
        router_temperature_end=0.5,
        router_temperature_schedule="step",
        balance_loss_enable=False,
        balance_loss_weight=1e-3,
        balance_loss_type="mse_uniform",
        router_entropy_penalty_enable=False,
        router_entropy_penalty_weight=1e-3,
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.dropout = dropout
        self.gate_type = gate_type
        self.gate_hidden_dim = gate_hidden_dim
        self.num_experts = max(int(num_experts), 1) if use_moe else 1
        self.use_moe = self.num_experts > 1
        self.balance_loss_enable = bool(balance_loss_enable)
        self.balance_loss_weight = float(balance_loss_weight)
        self.balance_loss_type = str(balance_loss_type).lower()
        self.router_entropy_penalty_enable = bool(router_entropy_penalty_enable)
        self.router_entropy_penalty_weight = float(router_entropy_penalty_weight)
        self.current_epoch = 1

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
                hidden_dim=gate_hidden_dim,
                temperature_enable=router_temperature_enable,
                temperature_start=router_temperature_start,
                temperature_mid=router_temperature_mid,
                temperature_end=router_temperature_end,
                temperature_schedule=router_temperature_schedule,
            )
        else:
            self.gate = None

    def set_current_epoch(self, epoch):
        self.current_epoch = max(int(epoch), 1)
        if self.gate is not None and hasattr(self.gate, "set_current_epoch"):
            self.gate.set_current_epoch(self.current_epoch)

    def _compute_balance_loss(self, gate_weights):
        if not self.balance_loss_enable or gate_weights.shape[-1] <= 1:
            return gate_weights.new_tensor(0.0)
        if self.balance_loss_type != "mse_uniform":
            raise ValueError(f"Unsupported balance loss type: {self.balance_loss_type}")

        flat_weights = gate_weights.reshape(-1, gate_weights.shape[-1])
        mean_usage = flat_weights.mean(dim=0)
        uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        return ((mean_usage - uniform) ** 2).sum()

    def _compute_router_entropy_penalty(self, gate_weights):
        if not self.router_entropy_penalty_enable or gate_weights.shape[-1] <= 1:
            return gate_weights.new_tensor(0.0)
        flat_weights = gate_weights.reshape(-1, gate_weights.shape[-1])
        return compute_expert_entropy(flat_weights).mean()

    def forward(self, x, return_routing=False):
        if not self.use_moe:
            output = self.net(x)
            if not return_routing:
                return output

            zero = output.new_tensor(0.0)
            gate_weights = torch.ones(
                *x.shape[:-1],
                1,
                device=x.device,
                dtype=output.dtype
            )
            routing_info = {
                "gate_weights": gate_weights,
                "top1_expert": torch.zeros(
                    *x.shape[:-1],
                    device=x.device,
                    dtype=torch.long
                ),
                "num_experts": 1,
                "gate_type": "single_expert",
                "router_temperature": output.new_tensor(1.0),
                "balance_loss": zero,
                "router_entropy_penalty": zero,
                "moe_aux_loss": zero,
                "mean_gate_margin": zero,
                "std_gate_margin": zero,
                "expert_output_cosine_mean": zero,
                "expert_output_cosine_std": zero,
            }
            return output, routing_info

        expert_outputs = [self.net(x)]
        expert_outputs.extend(expert(x) for expert in self.extra_experts)
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        gate_weights, gate_logits, router_temperature = self.gate(x)
        output = (expert_outputs * gate_weights.unsqueeze(dim=-1)).sum(dim=-2)

        if not return_routing:
            return output

        balance_loss = self._compute_balance_loss(gate_weights)
        router_entropy_penalty = self._compute_router_entropy_penalty(gate_weights)
        gate_margin = compute_gate_margin(gate_weights)
        expert_output_cosine_mean, expert_output_cosine_std = compute_expert_output_similarity(expert_outputs)
        moe_aux_loss = (
            self.balance_loss_weight * balance_loss
            + self.router_entropy_penalty_weight * router_entropy_penalty
        )
        routing_info = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "top1_expert": gate_weights.argmax(dim=-1),
            "num_experts": self.num_experts,
            "gate_type": self.gate_type,
            "router_temperature": output.new_tensor(float(router_temperature)),
            "balance_loss": balance_loss,
            "router_entropy_penalty": router_entropy_penalty,
            "moe_aux_loss": moe_aux_loss,
            "mean_gate_margin": gate_margin.mean(),
            "std_gate_margin": gate_margin.std(unbiased=False),
            "expert_output_cosine_mean": expert_output_cosine_mean,
            "expert_output_cosine_std": expert_output_cosine_std,
        }
        return output, routing_info

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
