from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_COORDINATE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _expert_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            column
            for column in frame.columns
            if re.fullmatch(r"expert_\d+", column)
        ],
        key=lambda name: int(name.split("_")[1]),
    )


def _layer_sort_key(name: Any) -> tuple[int, str]:
    text = str(name)
    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1)), text
    return math.inf, text


def _soft_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probabilities, eps, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def _gate_margin_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    num_experts = probabilities.shape[-1]
    if num_experts <= 1:
        return np.zeros(probabilities.shape[0], dtype=float)
    if num_experts == 2:
        return np.abs(probabilities[:, 0] - probabilities[:, 1])
    top2 = np.sort(probabilities, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _normalised_entropy(probabilities: np.ndarray) -> float:
    num_experts = probabilities.shape[-1]
    if num_experts <= 1:
        return 1.0
    mean_distribution = probabilities.mean(axis=0)
    entropy = float(_soft_entropy(mean_distribution[None, :])[0])
    return float(np.clip(entropy / math.log(num_experts), 0.0, 1.0))


def _l1_to_uniform(probabilities: np.ndarray) -> float:
    num_experts = probabilities.shape[-1]
    if num_experts <= 1:
        return 0.0
    uniform = np.full(num_experts, 1.0 / num_experts, dtype=float)
    return float(np.abs(probabilities.mean(axis=0) - uniform).sum())


def _balance_loss_to_uniform(probabilities: np.ndarray) -> float:
    num_experts = probabilities.shape[-1]
    if num_experts <= 1:
        return 0.0
    uniform = np.full(num_experts, 1.0 / num_experts, dtype=float)
    mean_distribution = probabilities.mean(axis=0)
    return float(((mean_distribution - uniform) ** 2).sum())


def _effective_expert_count(probabilities: np.ndarray) -> float:
    mean_distribution = probabilities.mean(axis=0)
    entropy = float(_soft_entropy(mean_distribution[None, :])[0])
    return float(math.exp(entropy))


def _js_divergence(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.clip(left / left.sum(), eps, 1.0)
    right = np.clip(right / right.sum(), eps, 1.0)
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(left * np.log(left / midpoint))
    kl_right = np.sum(right * np.log(right / midpoint))
    return float(0.5 * (kl_left + kl_right))


def _summarise_probability_block(probabilities: np.ndarray) -> dict[str, Any]:
    """Summarise how experts are used in one block of center spots.

    Interpretation:
    - `normalised_entropy` close to 1 means overall usage is well spread over experts.
    - `dominant_expert_fraction` close to 1 means one expert dominates the average routing mass.
    - `mean_weight_l1_to_uniform` and `top1_l1_to_uniform` close to 0 mean balanced usage.
    - `effective_expert_count` estimates how many experts are meaningfully active.
    """
    if probabilities.size == 0:
        raise ValueError("Cannot summarise an empty probability block.")

    num_spots, num_experts = probabilities.shape
    top1_index = probabilities.argmax(axis=1)
    top1_frequency = np.bincount(top1_index, minlength=num_experts) / max(num_spots, 1)
    mean_weights = probabilities.mean(axis=0)
    top1_entropy = float(_soft_entropy(top1_frequency[None, :])[0]) if num_experts > 1 else 0.0
    if num_experts <= 1:
        gate_margin = np.zeros(num_spots, dtype=float)
    elif num_experts == 2:
        gate_margin = np.abs(probabilities[:, 0] - probabilities[:, 1])
    else:
        top2 = np.sort(probabilities, axis=1)[:, -2:]
        gate_margin = top2[:, 1] - top2[:, 0]

    return {
        "num_center_spots": int(num_spots),
        "num_experts": int(num_experts),
        "normalised_entropy": _normalised_entropy(probabilities),
        "top1_normalised_entropy": float(
            np.clip(top1_entropy / math.log(num_experts), 0.0, 1.0)
        ) if num_experts > 1 else 1.0,
        "effective_expert_count": _effective_expert_count(probabilities),
        "dominant_expert_fraction": float(mean_weights.max()),
        "mean_weight_l1_to_uniform": _l1_to_uniform(probabilities),
        "balance_loss": _balance_loss_to_uniform(probabilities),
        "top1_l1_to_uniform": float(
            np.abs(top1_frequency - np.full(num_experts, 1.0 / num_experts)).sum()
        ) if num_experts > 1 else 0.0,
        "mean_spot_entropy": float(_soft_entropy(probabilities).mean()),
        "std_spot_entropy": float(_soft_entropy(probabilities).std()),
        "mean_gate_margin": float(gate_margin.mean()) if gate_margin.size else 0.0,
        "std_gate_margin": float(gate_margin.std()) if gate_margin.size else 0.0,
        "expert_mean_weights": mean_weights,
        "expert_top1_frequency": top1_frequency,
    }


def parse_sample_id(sample_id: Any) -> dict[str, Any]:
    text = str(sample_id)
    if "/" in text:
        slice_id, spot_id = text.split("/", 1)
    else:
        slice_id, spot_id = "default", text

    match = _COORDINATE_PATTERN.search(spot_id)
    x_value = y_value = np.nan
    if match:
        x_value = float(match.group(1))
        y_value = float(match.group(2))

    return {
        "sample_id": text,
        "slice_id": slice_id,
        "spot_id": spot_id,
        "x": x_value,
        "y": y_value,
    }


def _normalise_sample_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return dict(metadata)
    if isinstance(metadata, (tuple, list)) and len(metadata) == 2:
        return {"x": float(metadata[0]), "y": float(metadata[1])}
    raise TypeError("Sample metadata must be a mapping, a 2-tuple, or None.")


def resolve_sample_metadata(
    sample_id: Any,
    sample_metadata_resolver: Callable[[str], Any] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = parse_sample_id(sample_id)
    if sample_metadata_resolver is None:
        return metadata

    if callable(sample_metadata_resolver):
        resolved = sample_metadata_resolver(metadata["sample_id"])
    else:
        resolved = sample_metadata_resolver.get(metadata["sample_id"])
        if resolved is None:
            resolved = sample_metadata_resolver.get(metadata["spot_id"])

    extra = _normalise_sample_metadata(resolved)
    for key, value in extra.items():
        if value is not None:
            metadata[key] = value
    return metadata


def default_batch_adapter(
    batch: Sequence[Any],
    include_cell_information: bool = False,
    include_images: bool = False,
) -> dict[str, Any]:
    """Convert a dataloader batch into model inputs plus center-spot metadata."""
    if len(batch) == 4:
        return {
            "model_args": (batch[0], batch[2]),
            "sample_ids": batch[3],
            "targets": batch[1],
        }

    if len(batch) == 6 and torch.is_tensor(batch[0]) and batch[0].ndim == 4:
        return {
            "model_args": (batch[0], batch[1], batch[3]) if include_images else (batch[1], batch[3]),
            "sample_ids": batch[5],
            "targets": batch[2],
        }

    if len(batch) == 6:
        if include_cell_information:
            cell_info = torch.cat([batch[2][:, None, :], batch[4]], dim=1)
            model_args = (batch[0], batch[3], cell_info)
        else:
            model_args = (batch[0], batch[3])

        return {
            "model_args": model_args,
            "sample_ids": batch[5],
            "targets": batch[1],
        }

    raise ValueError(
        "Unsupported batch format. Pass a custom `batch_adapter` for this dataloader."
    )


def _layer_routing_infos(moe_info: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    layer_infos = moe_info.get("layer_routing_info")
    if isinstance(layer_infos, Sequence):
        valid_infos = [info for info in layer_infos if isinstance(info, Mapping)]
        if valid_infos:
            return valid_infos
    return [moe_info]


def _center_routing_arrays(
    routing_info: Mapping[str, Any],
    center_token_index: int = 0,
) -> dict[str, np.ndarray]:
    gate_weights = _to_numpy(routing_info["gate_weights"])
    if gate_weights.ndim >= 3:
        center_gate_weights = gate_weights[:, center_token_index, :]
    else:
        center_gate_weights = gate_weights

    return {
        "center_gate_weights": center_gate_weights,
        "center_entropy": _soft_entropy(center_gate_weights),
        "center_top1_expert": center_gate_weights.argmax(axis=-1).astype(int),
        "center_gate_margin": _gate_margin_from_probabilities(center_gate_weights),
    }


def _shared_routing_record_fields(
    routing_info: Mapping[str, Any],
    layer_index: int,
    num_layers: int,
) -> dict[str, float]:
    shared_fields = {
        "moe_layer_index": int(layer_index),
        "moe_num_layers": int(num_layers),
    }
    for key in (
        "router_temperature",
        "balance_loss",
        "router_entropy_penalty",
        "moe_aux_loss",
        "mean_gate_margin",
        "std_gate_margin",
        "expert_output_cosine_mean",
        "expert_output_cosine_std",
    ):
        if key in routing_info:
            shared_fields[key] = float(_to_numpy(routing_info[key]))
    return shared_fields


def _records_to_frame(records: list[dict[str, Any]], model_class_name: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_records(records)
    expert_columns = _expert_columns(frame)
    sort_columns = [column for column in ("moe_layer_index", "slice_id", "y", "x", "sample_id") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    frame.attrs["num_experts"] = len(expert_columns)
    frame.attrs["model_class"] = model_class_name
    return frame


def _collect_moe_activation_frames(
    model: Any,
    dataloader: Any,
    device: torch.device | None = None,
    batch_adapter: Callable[[Sequence[Any]], dict[str, Any]] | None = None,
    sample_metadata_resolver: Callable[[str], Any] | Mapping[str, Any] | None = None,
    include_cell_information: bool = False,
    include_images: bool = False,
    include_predictions: bool = True,
    include_targets: bool = False,
    max_batches: int | None = None,
    include_layerwise: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if batch_adapter is None:
        batch_adapter = lambda batch: default_batch_adapter(
            batch,
            include_cell_information=include_cell_information,
            include_images=include_images,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    was_training = model.training
    model.eval()

    final_records: list[dict[str, Any]] = []
    layer_records: dict[int, list[dict[str, Any]]] = {}
    unwrapped = unwrap_model(model)

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            adapted = batch_adapter(batch)
            model_args = tuple(
                tensor.to(device) if torch.is_tensor(tensor) else tensor
                for tensor in adapted["model_args"]
            )

            outputs = model(*model_args, return_moe_info=True)
            if not isinstance(outputs, Mapping) or "moe_info" not in outputs:
                raise RuntimeError(
                    "Model did not return MoE metadata. Make sure it supports `return_moe_info=True`."
                )

            predictions = outputs["predictions"]
            moe_info = outputs["moe_info"]
            layer_infos = _layer_routing_infos(moe_info)
            final_layer_index = len(layer_infos) - 1

            if include_predictions:
                predictions_np = _to_numpy(predictions)
            else:
                predictions_np = None

            if include_targets and adapted.get("targets") is not None:
                targets_np = _to_numpy(adapted["targets"])
            else:
                targets_np = None

            sample_ids = [str(sample_id) for sample_id in adapted["sample_ids"]]

            def append_records(
                target_records: list[dict[str, Any]],
                routing_info: Mapping[str, Any],
                layer_index: int,
            ) -> None:
                center_arrays = _center_routing_arrays(routing_info)
                center_gate_weights = center_arrays["center_gate_weights"]
                center_entropy = center_arrays["center_entropy"]
                center_top1 = center_arrays["center_top1_expert"]
                center_gate_margin = center_arrays["center_gate_margin"]
                shared_record_fields = _shared_routing_record_fields(
                    routing_info,
                    layer_index=layer_index,
                    num_layers=len(layer_infos),
                )

                for row_index, sample_id in enumerate(sample_ids):
                    metadata = resolve_sample_metadata(
                        sample_id,
                        sample_metadata_resolver=sample_metadata_resolver,
                    )
                    record = {
                        **metadata,
                        "batch_index": batch_index,
                        "batch_spot_index": row_index,
                        "top1_expert": int(center_top1[row_index]),
                        "top1_weight": float(center_gate_weights[row_index].max()),
                        "center_entropy": float(center_entropy[row_index]),
                        "gate_margin": float(center_gate_margin[row_index]),
                        "effective_expert_count_per_spot": float(
                            math.exp(float(center_entropy[row_index]))
                        ),
                        **shared_record_fields,
                    }

                    for expert_index, weight in enumerate(center_gate_weights[row_index]):
                        record[f"expert_{expert_index}"] = float(weight)

                    if predictions_np is not None:
                        for output_index, value in enumerate(np.atleast_1d(predictions_np[row_index])):
                            record[f"prediction_{output_index}"] = float(value)

                    if targets_np is not None:
                        for output_index, value in enumerate(np.atleast_1d(targets_np[row_index])):
                            record[f"target_{output_index}"] = float(value)

                    target_records.append(record)

            append_records(final_records, layer_infos[final_layer_index], final_layer_index)

            if include_layerwise:
                for layer_index, layer_info in enumerate(layer_infos):
                    layer_records.setdefault(layer_index, [])
                    append_records(layer_records[layer_index], layer_info, layer_index)

    if was_training:
        model.train()

    activation_frame = _records_to_frame(final_records, unwrapped.__class__.__name__)
    layer_activation_frames = {
        f"layer_{layer_index}": _records_to_frame(records, unwrapped.__class__.__name__)
        for layer_index, records in sorted(layer_records.items())
    }
    return activation_frame, layer_activation_frames


def collect_moe_activations(
    model: Any,
    dataloader: Any,
    device: torch.device | None = None,
    batch_adapter: Callable[[Sequence[Any]], dict[str, Any]] | None = None,
    sample_metadata_resolver: Callable[[str], Any] | Mapping[str, Any] | None = None,
    include_cell_information: bool = False,
    include_images: bool = False,
    include_predictions: bool = True,
    include_targets: bool = False,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Run one forward pass over a loader and collect center-spot expert weights.

    The returned dataframe is the main analysis table. Each row is one center spot,
    with one `expert_i` column per expert plus optional prediction/target columns.
    """
    activation_frame, _ = _collect_moe_activation_frames(
        model=model,
        dataloader=dataloader,
        device=device,
        batch_adapter=batch_adapter,
        sample_metadata_resolver=sample_metadata_resolver,
        include_cell_information=include_cell_information,
        include_images=include_images,
        include_predictions=include_predictions,
        include_targets=include_targets,
        max_batches=max_batches,
        include_layerwise=False,
    )
    return activation_frame


def collect_layerwise_moe_activations(
    model: Any,
    dataloader: Any,
    device: torch.device | None = None,
    batch_adapter: Callable[[Sequence[Any]], dict[str, Any]] | None = None,
    sample_metadata_resolver: Callable[[str], Any] | Mapping[str, Any] | None = None,
    include_cell_information: bool = False,
    include_images: bool = False,
    include_predictions: bool = True,
    include_targets: bool = False,
    max_batches: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Collect one activation dataframe per MoE layer.

    The returned mapping uses keys like `layer_0`, `layer_1`, ... and keeps the
    same per-spot schema as `collect_moe_activations(...)`.
    """
    _, layer_activation_frames = _collect_moe_activation_frames(
        model=model,
        dataloader=dataloader,
        device=device,
        batch_adapter=batch_adapter,
        sample_metadata_resolver=sample_metadata_resolver,
        include_cell_information=include_cell_information,
        include_images=include_images,
        include_predictions=include_predictions,
        include_targets=include_targets,
        max_batches=max_batches,
        include_layerwise=True,
    )
    return layer_activation_frames


def compute_grouped_usage_metrics(
    activation_frame: pd.DataFrame,
    groupby: str | list[str],
) -> pd.DataFrame:
    """Compute load-balance and collapse indicators for each batch/slice/region group."""
    if activation_frame.empty:
        return pd.DataFrame()

    expert_columns = _expert_columns(activation_frame)
    if not expert_columns:
        raise ValueError("No expert columns were found in the activation dataframe.")

    groups = activation_frame.groupby(groupby, dropna=False)
    records: list[dict[str, Any]] = []
    for group_key, group_frame in groups:
        probabilities = group_frame[expert_columns].to_numpy(dtype=float)
        summary = _summarise_probability_block(probabilities)
        group_record = {}

        if isinstance(group_key, tuple):
            group_names = list(groupby) if isinstance(groupby, list) else [groupby]
            group_record.update(dict(zip(group_names, group_key)))
        else:
            group_name = groupby if isinstance(groupby, str) else "_".join(groupby)
            group_record[group_name] = group_key

        for expert_index, value in enumerate(summary["expert_mean_weights"]):
            group_record[f"expert_{expert_index}_mean_weight"] = float(value)
        for expert_index, value in enumerate(summary["expert_top1_frequency"]):
            group_record[f"expert_{expert_index}_top1_frequency"] = float(value)

        group_record.update(
            {
                "num_center_spots": summary["num_center_spots"],
                "normalised_entropy": summary["normalised_entropy"],
                "top1_normalised_entropy": summary["top1_normalised_entropy"],
                "effective_expert_count": summary["effective_expert_count"],
                "dominant_expert_fraction": summary["dominant_expert_fraction"],
                "mean_weight_l1_to_uniform": summary["mean_weight_l1_to_uniform"],
                "balance_loss": summary["balance_loss"],
                "top1_l1_to_uniform": summary["top1_l1_to_uniform"],
                "mean_spot_entropy": summary["mean_spot_entropy"],
                "std_spot_entropy": summary["std_spot_entropy"],
                "mean_gate_margin": summary["mean_gate_margin"],
                "std_gate_margin": summary["std_gate_margin"],
            }
        )
        for column in (
            "router_temperature",
            "balance_loss",
            "router_entropy_penalty",
            "expert_output_cosine_mean",
            "expert_output_cosine_std",
        ):
            if column in group_frame.columns:
                group_record[column] = float(group_frame[column].mean())
        records.append(group_record)

    return pd.DataFrame.from_records(records)


def compare_usage_patterns(
    activation_frame: pd.DataFrame,
    groupby: str,
) -> pd.DataFrame:
    """Pairwise compare expert usage between slices or spatial regions.

    Larger `l1_distance` / `js_divergence` values mean the groups use experts more
    differently, which is a useful sign of niche-specific specialisation.
    """
    grouped = compute_grouped_usage_metrics(activation_frame, groupby=groupby)
    if grouped.empty:
        return grouped

    expert_columns = sorted(
        [column for column in grouped.columns if column.endswith("_mean_weight")],
        key=lambda name: int(name.split("_")[1]),
    )
    if grouped.shape[0] < 2:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for left_index in range(grouped.shape[0]):
        for right_index in range(left_index + 1, grouped.shape[0]):
            left_name = grouped.iloc[left_index][groupby]
            right_name = grouped.iloc[right_index][groupby]
            left_distribution = grouped.iloc[left_index][expert_columns].to_numpy(dtype=float)
            right_distribution = grouped.iloc[right_index][expert_columns].to_numpy(dtype=float)

            records.append(
                {
                    groupby + "_a": left_name,
                    groupby + "_b": right_name,
                    "l1_distance": float(np.abs(left_distribution - right_distribution).sum()),
                    "js_divergence": _js_divergence(left_distribution, right_distribution),
                }
            )

    return pd.DataFrame.from_records(records)


def assign_spatial_regions(
    activation_frame: pd.DataFrame,
    bins: int = 2,
    x_column: str = "x",
    y_column: str = "y",
    region_column: str = "spatial_region",
) -> pd.DataFrame:
    """Split each slice into coarse spatial regions using quantile bins."""
    if activation_frame.empty:
        return activation_frame.copy()
    if x_column not in activation_frame or y_column not in activation_frame:
        raise ValueError("Spatial columns are required to define spatial regions.")

    frame = activation_frame.copy()
    valid_mask = frame[x_column].notna() & frame[y_column].notna()
    if not valid_mask.any():
        raise ValueError("No valid coordinates are available to define spatial regions.")

    frame[region_column] = pd.Series(index=frame.index, dtype=object)

    if "slice_id" in frame.columns:
        grouped_index = frame.loc[valid_mask].groupby("slice_id", dropna=False).groups
    else:
        grouped_index = {"default": frame.index[valid_mask]}

    for slice_id, index in grouped_index.items():
        x_bins = pd.qcut(
            frame.loc[index, x_column].rank(method="first"),
            q=min(bins, len(index)),
            labels=False,
            duplicates="drop",
        )
        y_bins = pd.qcut(
            frame.loc[index, y_column].rank(method="first"),
            q=min(bins, len(index)),
            labels=False,
            duplicates="drop",
        )
        frame.loc[index, region_column] = [
            f"{slice_id}|x{int(x_value)}_y{int(y_value)}"
            for x_value, y_value in zip(x_bins.astype(int), y_bins.astype(int))
        ]
    return frame


def compute_expert_usage_metrics(
    activation_frame: pd.DataFrame,
    add_spatial_regions: bool = True,
    spatial_region_bins: int = 2,
) -> dict[str, Any]:
    """Compute overall, batch-level, slice-level, and region-level MoE metrics."""
    if activation_frame.empty:
        return {
            "activation_frame": activation_frame.copy(),
            "overall": {},
            "expert_summary": pd.DataFrame(),
            "batch_summary": pd.DataFrame(),
            "slice_summary": pd.DataFrame(),
            "region_summary": pd.DataFrame(),
            "slice_differences": pd.DataFrame(),
            "region_differences": pd.DataFrame(),
        }

    expert_columns = _expert_columns(activation_frame)
    probabilities = activation_frame[expert_columns].to_numpy(dtype=float)
    summary = _summarise_probability_block(probabilities)

    expert_summary = pd.DataFrame(
        {
            "expert": list(range(len(expert_columns))),
            "average_activation_weight": summary["expert_mean_weights"],
            "top1_selection_frequency": summary["expert_top1_frequency"],
        }
    )

    batch_summary = compute_grouped_usage_metrics(activation_frame, groupby="batch_index")
    slice_summary = compute_grouped_usage_metrics(activation_frame, groupby="slice_id")
    slice_differences = compare_usage_patterns(activation_frame, groupby="slice_id")

    region_summary = pd.DataFrame()
    region_differences = pd.DataFrame()
    output_frame = activation_frame
    if add_spatial_regions and activation_frame["x"].notna().any() and activation_frame["y"].notna().any():
        output_frame = assign_spatial_regions(activation_frame, bins=spatial_region_bins)
        region_summary = compute_grouped_usage_metrics(output_frame, groupby="spatial_region")
        region_differences = compare_usage_patterns(output_frame, groupby="spatial_region")

    overall = {
        "num_center_spots": summary["num_center_spots"],
        "num_experts": summary["num_experts"],
        "usage_entropy_normalised": summary["normalised_entropy"],
        "top1_entropy_normalised": summary["top1_normalised_entropy"],
        "effective_expert_count": summary["effective_expert_count"],
        "dominant_expert_fraction": summary["dominant_expert_fraction"],
        "mean_weight_l1_to_uniform": summary["mean_weight_l1_to_uniform"],
        "balance_loss": summary["balance_loss"],
        "top1_l1_to_uniform": summary["top1_l1_to_uniform"],
        "mean_spot_entropy": summary["mean_spot_entropy"],
        "std_spot_entropy": summary["std_spot_entropy"],
        "mean_gate_margin": summary["mean_gate_margin"],
        "std_gate_margin": summary["std_gate_margin"],
    }
    for column in (
        "router_temperature",
        "balance_loss",
        "router_entropy_penalty",
        "expert_output_cosine_mean",
        "expert_output_cosine_std",
    ):
        if column in output_frame.columns:
            overall[column] = float(output_frame[column].mean())
    for column in ("moe_layer_index", "moe_num_layers"):
        if column in output_frame.columns and not output_frame.empty:
            overall[column] = int(output_frame[column].iloc[0])

    return {
        "activation_frame": output_frame,
        "overall": overall,
        "expert_summary": expert_summary,
        "batch_summary": batch_summary,
        "slice_summary": slice_summary,
        "region_summary": region_summary,
        "slice_differences": slice_differences,
        "region_differences": region_differences,
    }


def compute_layerwise_expert_usage_metrics(
    layer_activation_frames: Mapping[str, pd.DataFrame],
    add_spatial_regions: bool = True,
    spatial_region_bins: int = 2,
) -> dict[str, Any]:
    """Compute expert-usage summaries for every MoE layer independently."""
    layer_results: dict[str, dict[str, Any]] = {}
    summary_records: list[dict[str, Any]] = []

    for layer_name in sorted(layer_activation_frames, key=_layer_sort_key):
        layer_frame = layer_activation_frames[layer_name]
        metrics = compute_expert_usage_metrics(
            activation_frame=layer_frame,
            add_spatial_regions=add_spatial_regions,
            spatial_region_bins=spatial_region_bins,
        )
        layer_results[layer_name] = metrics

        summary_record = {"layer_name": layer_name}
        summary_record.update(metrics.get("overall", {}))
        if isinstance(layer_frame, pd.DataFrame) and not layer_frame.empty:
            if "moe_layer_index" in layer_frame.columns:
                summary_record["moe_layer_index"] = int(layer_frame["moe_layer_index"].iloc[0])
            if "moe_num_layers" in layer_frame.columns:
                summary_record["moe_num_layers"] = int(layer_frame["moe_num_layers"].iloc[0])
        summary_records.append(summary_record)

    layer_overall_summary = pd.DataFrame.from_records(summary_records)
    if not layer_overall_summary.empty:
        sort_columns = [
            column
            for column in ("moe_layer_index", "layer_name")
            if column in layer_overall_summary.columns
        ]
        if sort_columns:
            layer_overall_summary = (
                layer_overall_summary.sort_values(sort_columns).reset_index(drop=True)
            )

    return {
        "layer_results": layer_results,
        "layer_overall_summary": layer_overall_summary,
    }


def analyze_moe_routing(
    model: Any,
    dataloader: Any,
    device: torch.device | None = None,
    batch_adapter: Callable[[Sequence[Any]], dict[str, Any]] | None = None,
    sample_metadata_resolver: Callable[[str], Any] | Mapping[str, Any] | None = None,
    include_cell_information: bool = False,
    include_images: bool = False,
    include_predictions: bool = True,
    include_targets: bool = False,
    max_batches: int | None = None,
    add_spatial_regions: bool = True,
    spatial_region_bins: int = 2,
) -> dict[str, Any]:
    activation_frame, layer_activation_frames = _collect_moe_activation_frames(
        model=model,
        dataloader=dataloader,
        device=device,
        batch_adapter=batch_adapter,
        sample_metadata_resolver=sample_metadata_resolver,
        include_cell_information=include_cell_information,
        include_images=include_images,
        include_predictions=include_predictions,
        include_targets=include_targets,
        max_batches=max_batches,
        include_layerwise=True,
    )
    analysis_results = compute_expert_usage_metrics(
        activation_frame=activation_frame,
        add_spatial_regions=add_spatial_regions,
        spatial_region_bins=spatial_region_bins,
    )
    layerwise_results = compute_layerwise_expert_usage_metrics(
        layer_activation_frames=layer_activation_frames,
        add_spatial_regions=add_spatial_regions,
        spatial_region_bins=spatial_region_bins,
    )
    analysis_results["layer_activation_frames"] = layer_activation_frames
    analysis_results.update(layerwise_results)
    return analysis_results


def summarize_epoch_trajectory(epoch_activation_frames: Mapping[Any, pd.DataFrame]) -> pd.DataFrame:
    """Track whether expert usage becomes more stable or more specialised over time."""
    records: list[dict[str, Any]] = []
    for epoch, frame in epoch_activation_frames.items():
        metrics = compute_expert_usage_metrics(frame, add_spatial_regions=False)
        records.append({"epoch": epoch, **metrics["overall"]})
    return pd.DataFrame.from_records(records).sort_values("epoch").reset_index(drop=True)


def _slice_frame(activation_frame: pd.DataFrame, slice_id: str | None) -> pd.DataFrame:
    if activation_frame.empty:
        raise ValueError("Activation dataframe is empty.")
    if slice_id is None:
        unique_slices = activation_frame["slice_id"].dropna().unique().tolist()
        if len(unique_slices) == 1:
            slice_id = unique_slices[0]
        else:
            raise ValueError("Multiple slices are present. Please provide `slice_id`.")
    subset = activation_frame.loc[activation_frame["slice_id"] == slice_id].copy()
    if subset.empty:
        raise ValueError(f"Slice '{slice_id}' was not found in the activation dataframe.")
    return subset


def plot_slice_activation_heatmap(
    activation_frame: pd.DataFrame,
    slice_id: str | None = None,
    sort_by: Sequence[str] = ("y", "x", "sample_id"),
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of center-spot expert weights for one slice."""
    subset = _slice_frame(activation_frame, slice_id)
    expert_columns = _expert_columns(subset)
    available_sort = [column for column in sort_by if column in subset.columns]
    if available_sort:
        subset = subset.sort_values(available_sort)

    matrix = subset[expert_columns].to_numpy(dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Center spot")
    ax.set_xticks(range(len(expert_columns)))
    ax.set_xticklabels([column.replace("expert_", "E") for column in expert_columns])
    ax.set_title(f"MoE activations for slice {subset['slice_id'].iloc[0]}")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Gate weight")
    return ax


def plot_expert_spatial_heatmap(
    activation_frame: pd.DataFrame,
    expert_index: int,
    slice_id: str | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    point_size: float = 24.0,
    invert_y: bool = True,
) -> plt.Axes:
    """Scatter-plot one expert's activation over tissue coordinates."""
    subset = _slice_frame(activation_frame, slice_id)
    if subset["x"].isna().all() or subset["y"].isna().all():
        raise ValueError("This slice does not have spatial coordinates for plotting.")

    expert_column = f"expert_{expert_index}"
    if expert_column not in subset.columns:
        raise ValueError(f"Expert column '{expert_column}' was not found.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    scatter = ax.scatter(
        subset["x"],
        subset["y"],
        c=subset[expert_column],
        cmap=cmap,
        s=point_size,
        linewidths=0,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Expert {expert_index} spatial activation")
    if invert_y:
        ax.invert_yaxis()
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Gate weight")
    return ax


def plot_all_experts_spatial_grid(
    activation_frame: pd.DataFrame,
    slice_id: str | None = None,
    columns: int = 4,
    cmap: str = "viridis",
    point_size: float = 20.0,
) -> tuple[plt.Figure, np.ndarray]:
    """Small multiples view of spatial expert preference across a whole slice."""
    subset = _slice_frame(activation_frame, slice_id)
    expert_columns = _expert_columns(subset)
    num_experts = len(expert_columns)
    rows = int(math.ceil(num_experts / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)

    for expert_index in range(rows * columns):
        axis = axes.flat[expert_index]
        if expert_index >= num_experts:
            axis.axis("off")
            continue
        plot_expert_spatial_heatmap(
            subset,
            expert_index=expert_index,
            slice_id=subset["slice_id"].iloc[0],
            ax=axis,
            cmap=cmap,
            point_size=point_size,
        )
    figure.tight_layout()
    return figure, axes


def plot_center_spot_activation_bar(
    activation_frame: pd.DataFrame,
    sample_id: str | None = None,
    row_index: int | None = None,
    ax: plt.Axes | None = None,
    color: str = "#4c72b0",
) -> plt.Axes:
    """Bar chart of one center spot's expert routing vector."""
    if activation_frame.empty:
        raise ValueError("Activation dataframe is empty.")
    if sample_id is None and row_index is None:
        raise ValueError("Pass either `sample_id` or `row_index`.")

    if sample_id is not None:
        subset = activation_frame.loc[activation_frame["sample_id"] == sample_id]
        if subset.empty:
            raise ValueError(f"Sample '{sample_id}' was not found.")
        record = subset.iloc[0]
    else:
        record = activation_frame.iloc[int(row_index)]

    expert_columns = _expert_columns(activation_frame)
    values = record[expert_columns].to_numpy(dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.bar(range(len(values)), values, color=color)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Gate weight")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([column.replace("expert_", "E") for column in expert_columns])
    ax.set_title(f"Center-spot routing for {record['sample_id']}")
    return ax


def save_moe_analysis_tables(
    analysis_results: Mapping[str, Any],
    output_dir: str | Path,
    prefix: str = "moe",
) -> dict[str, Path]:
    """Save the activation table and metric tables so they can be reloaded later."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    table_keys = (
        "activation_frame",
        "expert_summary",
        "batch_summary",
        "slice_summary",
        "region_summary",
        "slice_differences",
        "region_differences",
    )

    def save_frame(path_key: str, file_name: str, value: Any) -> None:
        if isinstance(value, pd.DataFrame) and not value.empty:
            path = output_path / file_name
            value.to_csv(path, index=False)
            saved_paths[path_key] = path

    for key in (
        "activation_frame",
        "expert_summary",
        "batch_summary",
        "slice_summary",
        "region_summary",
        "slice_differences",
        "region_differences",
    ):
        save_frame(key, f"{prefix}_{key}.csv", analysis_results.get(key))

    overall = analysis_results.get("overall")
    if overall:
        path = output_path / f"{prefix}_overall_metrics.json"
        pd.Series(overall).to_json(path, indent=2)
        saved_paths["overall"] = path

    save_frame(
        "layer_overall_summary",
        f"{prefix}_layer_overall_summary.csv",
        analysis_results.get("layer_overall_summary"),
    )

    layer_results = analysis_results.get("layer_results")
    saved_layer_activation = set()
    if isinstance(layer_results, Mapping):
        for layer_name in sorted(layer_results, key=_layer_sort_key):
            layer_metrics = layer_results.get(layer_name)
            if not isinstance(layer_metrics, Mapping):
                continue

            for key in table_keys:
                value = layer_metrics.get(key)
                path_key = f"{layer_name}_{key}"
                save_frame(path_key, f"{prefix}_{layer_name}_{key}.csv", value)
                if key == "activation_frame" and path_key in saved_paths:
                    saved_layer_activation.add(layer_name)

            layer_overall = layer_metrics.get("overall")
            if layer_overall:
                path = output_path / f"{prefix}_{layer_name}_overall_metrics.json"
                pd.Series(layer_overall).to_json(path, indent=2)
                saved_paths[f"{layer_name}_overall"] = path

    layer_activation_frames = analysis_results.get("layer_activation_frames")
    if isinstance(layer_activation_frames, Mapping):
        for layer_name in sorted(layer_activation_frames, key=_layer_sort_key):
            if layer_name in saved_layer_activation:
                continue
            save_frame(
                f"{layer_name}_activation_frame",
                f"{prefix}_{layer_name}_activation_frame.csv",
                layer_activation_frames.get(layer_name),
            )

    return saved_paths
