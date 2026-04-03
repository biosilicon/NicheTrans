from __future__ import annotations

import inspect
from typing import Any


MODEL_HPARAM_KEYS = (
    "noise_rate",
    "dropout_rate",
    "use_moe_ffn",
    "num_experts",
    "moe_gate_hidden_dim",
    "moe_gate_type",
    "ffn_mult",
    "moe_router_temperature_enable",
    "moe_router_temperature_start",
    "moe_router_temperature_mid",
    "moe_router_temperature_end",
    "moe_router_temperature_schedule",
    "moe_balance_loss_enable",
    "moe_balance_loss_weight",
    "moe_balance_loss_type",
    "moe_router_entropy_penalty_enable",
    "moe_router_entropy_penalty_weight",
)


def model_kwargs_from_args(model_cls: type, args: Any, **overrides: Any) -> dict[str, Any]:
    parameters = inspect.signature(model_cls.__init__).parameters
    kwargs: dict[str, Any] = {}

    for key in MODEL_HPARAM_KEYS:
        if key in parameters and hasattr(args, key):
            kwargs[key] = getattr(args, key)

    kwargs.update(overrides)
    return kwargs


def build_model_from_args(
    model_cls: type,
    args: Any,
    source_length: int,
    target_length: int,
    **overrides: Any,
) -> Any:
    kwargs = model_kwargs_from_args(
        model_cls,
        args,
        source_length=source_length,
        target_length=target_length,
        **overrides,
    )
    return model_cls(**kwargs)
