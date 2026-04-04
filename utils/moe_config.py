from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable


_MISSING = object()


def parse_literal_value(raw: Any) -> Any:
    """Parse CLI / JSON-like scalar text into Python values.

    Supports booleans, nulls, numbers, lists, tuples, and dicts while keeping
    ordinary strings such as `softmax` or `step` untouched.
    """
    if not isinstance(raw, str):
        return raw

    text = raw.strip()
    lowered = text.lower()

    if lowered in {"true", "yes", "y", "on"}:
        return True
    if lowered in {"false", "no", "n", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        if text.startswith("0") and len(text) > 1 and text[1].isdigit() and "." not in text:
            raise ValueError
        return int(text)
    except ValueError:
        pass

    try:
        value = float(text)
        if math.isfinite(value):
            return value
    except ValueError:
        pass

    if text and text[0] in "[{(":
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue

    return text


def split_top_level_commas(text: str) -> list[str]:
    """Split a comma-separated string while respecting nested []/()/{} groups."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote_char = ""
    escaped = False

    for char in text:
        if quote_char:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = ""
            continue

        if char in {"'", '"'}:
            quote_char = char
            current.append(char)
            continue

        if char in "[{(":
            depth += 1
            current.append(char)
            continue

        if char in "]})":
            depth = max(depth - 1, 0)
            current.append(char)
            continue

        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _coerce_bool(value: Any) -> bool:
    parsed = parse_literal_value(value)
    if isinstance(parsed, bool):
        return parsed
    raise ValueError(f"Expected a boolean-compatible value, got {value!r}.")


def _coerce_str(value: Any) -> str:
    parsed = parse_literal_value(value)
    return str(parsed)


def _coerce_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    parsed = parse_literal_value(value)
    number = int(parsed)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return number


def _coerce_positive_float(value: Any, name: str, minimum: float = 0.0) -> float:
    parsed = parse_literal_value(value)
    number = float(parsed)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return number


def _coerce_gate_hidden_dim(value: Any) -> int | None:
    parsed = parse_literal_value(value)
    if parsed in (None, 0):
        return None
    number = int(parsed)
    if number < 0:
        raise ValueError(f"moe_gate_hidden_dim must be >= 0, got {value!r}.")
    return number


def expand_layerwise_value(
    value: Any,
    num_layers: int,
    name: str,
    normalizer: Callable[[Any], Any] | None = None,
) -> list[Any]:
    """Expand scalar/list/dict config values into one value per layer."""
    total_layers = max(int(num_layers), 1)

    def normalize(item: Any) -> Any:
        return normalizer(item) if normalizer is not None else item

    if isinstance(value, Mapping):
        default = _MISSING
        for key in ("default", "all", "shared"):
            if key in value:
                default = value[key]
                break

        expanded = []
        for layer_index in range(total_layers):
            item = _MISSING
            for key in (layer_index, str(layer_index), f"layer_{layer_index}"):
                if key in value:
                    item = value[key]
                    break
            if item is _MISSING:
                if default is _MISSING:
                    raise ValueError(
                        f"{name} mapping is missing layer {layer_index} and does not define a default."
                    )
                item = default
            expanded.append(normalize(item))
        return expanded

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        expanded = list(value)
        if len(expanded) == 1 and total_layers > 1:
            expanded = expanded * total_layers
        if len(expanded) != total_layers:
            raise ValueError(
                f"{name} expects {total_layers} values, got {len(expanded)}."
            )
        return [normalize(item) for item in expanded]

    return [normalize(value) for _ in range(total_layers)]


def resolve_moe_layer_configs(
    *,
    num_layers: int,
    num_experts: Any,
    gate_hidden_dim: Any,
    mult: Any,
    use_moe: Any,
    gate_type: Any,
    router_temperature_enable: Any,
    router_temperature_start: Any,
    router_temperature_mid: Any,
    router_temperature_end: Any,
    router_temperature_schedule: Any,
    balance_loss_enable: Any,
    balance_loss_weight: Any,
    balance_loss_type: Any,
    router_entropy_penalty_enable: Any,
    router_entropy_penalty_weight: Any,
) -> list[dict[str, Any]]:
    """Resolve MoE hyper-parameters into per-layer block configs."""
    total_layers = max(int(num_layers), 1)

    layer_num_experts = expand_layerwise_value(
        num_experts,
        total_layers,
        "num_experts",
        normalizer=lambda item: _coerce_positive_int(item, "num_experts", minimum=1),
    )
    layer_gate_hidden_dim = expand_layerwise_value(
        gate_hidden_dim,
        total_layers,
        "moe_gate_hidden_dim",
        normalizer=_coerce_gate_hidden_dim,
    )
    layer_mult = expand_layerwise_value(
        mult,
        total_layers,
        "ffn_mult",
        normalizer=lambda item: _coerce_positive_int(item, "ffn_mult", minimum=1),
    )
    layer_use_moe = expand_layerwise_value(
        use_moe,
        total_layers,
        "use_moe_ffn",
        normalizer=_coerce_bool,
    )
    layer_gate_type = expand_layerwise_value(
        gate_type,
        total_layers,
        "moe_gate_type",
        normalizer=_coerce_str,
    )
    layer_temperature_enable = expand_layerwise_value(
        router_temperature_enable,
        total_layers,
        "moe_router_temperature_enable",
        normalizer=_coerce_bool,
    )
    layer_temperature_start = expand_layerwise_value(
        router_temperature_start,
        total_layers,
        "moe_router_temperature_start",
        normalizer=lambda item: _coerce_positive_float(item, "moe_router_temperature_start"),
    )
    layer_temperature_mid = expand_layerwise_value(
        router_temperature_mid,
        total_layers,
        "moe_router_temperature_mid",
        normalizer=lambda item: _coerce_positive_float(item, "moe_router_temperature_mid"),
    )
    layer_temperature_end = expand_layerwise_value(
        router_temperature_end,
        total_layers,
        "moe_router_temperature_end",
        normalizer=lambda item: _coerce_positive_float(item, "moe_router_temperature_end"),
    )
    layer_temperature_schedule = expand_layerwise_value(
        router_temperature_schedule,
        total_layers,
        "moe_router_temperature_schedule",
        normalizer=_coerce_str,
    )
    layer_balance_enable = expand_layerwise_value(
        balance_loss_enable,
        total_layers,
        "moe_balance_loss_enable",
        normalizer=_coerce_bool,
    )
    layer_balance_weight = expand_layerwise_value(
        balance_loss_weight,
        total_layers,
        "moe_balance_loss_weight",
        normalizer=lambda item: _coerce_positive_float(item, "moe_balance_loss_weight"),
    )
    layer_balance_type = expand_layerwise_value(
        balance_loss_type,
        total_layers,
        "moe_balance_loss_type",
        normalizer=_coerce_str,
    )
    layer_entropy_enable = expand_layerwise_value(
        router_entropy_penalty_enable,
        total_layers,
        "moe_router_entropy_penalty_enable",
        normalizer=_coerce_bool,
    )
    layer_entropy_weight = expand_layerwise_value(
        router_entropy_penalty_weight,
        total_layers,
        "moe_router_entropy_penalty_weight",
        normalizer=lambda item: _coerce_positive_float(item, "moe_router_entropy_penalty_weight"),
    )

    layer_configs: list[dict[str, Any]] = []
    for layer_index in range(total_layers):
        enabled = bool(layer_use_moe[layer_index])
        experts = int(layer_num_experts[layer_index])
        if not enabled:
            experts = 1

        layer_configs.append(
            {
                "layer_index": layer_index,
                "use_moe": enabled,
                "num_experts": experts,
                "gate_hidden_dim": layer_gate_hidden_dim[layer_index],
                "mult": int(layer_mult[layer_index]),
                "gate_type": layer_gate_type[layer_index],
                "router_temperature_enable": bool(layer_temperature_enable[layer_index]),
                "router_temperature_start": float(layer_temperature_start[layer_index]),
                "router_temperature_mid": float(layer_temperature_mid[layer_index]),
                "router_temperature_end": float(layer_temperature_end[layer_index]),
                "router_temperature_schedule": layer_temperature_schedule[layer_index],
                "balance_loss_enable": bool(layer_balance_enable[layer_index]),
                "balance_loss_weight": float(layer_balance_weight[layer_index]),
                "balance_loss_type": layer_balance_type[layer_index],
                "router_entropy_penalty_enable": bool(layer_entropy_enable[layer_index]),
                "router_entropy_penalty_weight": float(layer_entropy_weight[layer_index]),
            }
        )

    return layer_configs
