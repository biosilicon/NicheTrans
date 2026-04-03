from __future__ import annotations

from typing import Any, Mapping

import torch


MOE_TRAINING_LOG_KEYS = (
    "task_loss",
    "moe_aux_loss",
    "router_temperature",
    "balance_loss",
    "router_entropy_penalty",
    "mean_gate_margin",
    "std_gate_margin",
    "expert_output_cosine_mean",
    "expert_output_cosine_std",
)


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def prepare_moe_epoch(model: Any, epoch: int | None = None) -> int:
    unwrapped = unwrap_model(model)
    if hasattr(unwrapped, "set_current_epoch"):
        if epoch is None:
            epoch = getattr(unwrapped, "current_epoch", 0) + 1
        unwrapped.set_current_epoch(epoch)
        return int(epoch)

    ffn_module = getattr(unwrapped, "ffn_omic", None)
    if ffn_module is None or not hasattr(ffn_module, "set_current_epoch"):
        return 1

    if epoch is None:
        epoch = getattr(ffn_module, "current_epoch", 0) + 1

    ffn_module.set_current_epoch(epoch)
    return int(epoch)


def unpack_model_outputs(outputs: Any) -> tuple[Any, Mapping[str, Any] | None]:
    if isinstance(outputs, Mapping):
        return outputs.get("predictions"), outputs.get("moe_info")
    return outputs, None


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def combine_task_and_moe_loss(
    task_loss: torch.Tensor,
    moe_info: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics = {key: 0.0 for key in MOE_TRAINING_LOG_KEYS}
    metrics["task_loss"] = _to_float(task_loss)

    if not isinstance(moe_info, Mapping):
        return task_loss, metrics

    total_loss = task_loss
    if torch.is_tensor(moe_info.get("moe_aux_loss")):
        total_loss = total_loss + moe_info["moe_aux_loss"]
    metrics["moe_aux_loss"] = _to_float(moe_info.get("moe_aux_loss"))

    for key in MOE_TRAINING_LOG_KEYS:
        if key in ("task_loss", "moe_aux_loss"):
            continue
        if key in moe_info:
            metrics[key] = _to_float(moe_info[key])

    return total_loss, metrics


def update_metric_totals(
    metric_totals: dict[str, float],
    metrics: Mapping[str, float],
    batch_size: int,
) -> None:
    for key, value in metrics.items():
        metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * batch_size


def finalize_metric_totals(
    metric_totals: Mapping[str, float],
    total_count: int,
) -> dict[str, float]:
    if total_count <= 0:
        return {key: 0.0 for key in metric_totals}
    return {
        key: float(value) / float(total_count)
        for key, value in metric_totals.items()
    }
