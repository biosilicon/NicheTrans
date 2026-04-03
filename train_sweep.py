#!/usr/bin/env python3
"""Run hyper-parameter sweeps for NicheTrans datasets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


COMMON_METRIC_KEYS = [
    "pearson_mean",
    "pearson_std",
    "spearman_mean",
    "spearman_std",
    "rmse_mean",
    "rmse_std",
    "mean_auc",
    "tau_auc",
    "tau_sensitivity",
    "tau_specificity",
    "plaque_auc",
    "plaque_sensitivity",
    "plaque_specificity",
    "mean_auroc",
]


def parse_scalar(raw: str) -> Any:
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

    return text


def normalize_key(key: str) -> str:
    return key.strip().replace("-", "_")


def parse_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    return normalize_key(key), parse_scalar(value)


def parse_grid_assignment(raw: str) -> tuple[str, list[Any]]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=V1,V2,..., got: {raw}")
    key, value = raw.split("=", 1)
    values = [parse_scalar(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Grid for '{key}' is empty.")
    return normalize_key(key), values


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Grid config JSON must be an object.")
    return data


def normalize_grid_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def namespace_from_dict(values: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**values)


def serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, (list, tuple)):
        return [serialise(item) for item in value]
    if isinstance(value, dict):
        return {key: serialise(item) for key, item in value.items()}
    return value


def stable_json_dump(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialise(payload), handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialise(payload), sort_keys=True) + "\n")


def ensure_metric_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    merged = {key: "" for key in COMMON_METRIC_KEYS}
    merged.update({key: serialise(value) for key, value in metrics.items()})
    return merged


def save_summary_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: serialise(row.get(key, "")) for key in fieldnames})


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def torch_state_dict(model: Any) -> dict[str, Any]:
    return unwrap_model(model).state_dict()


def mean_or_nan(values: Any) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.nanmean(array))


def std_or_nan(values: Any) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.nanstd(array))


def evaluate_regression(
    model: Any,
    loader: Any,
    source_idx: int,
    target_idx: int,
    neighbor_idx: int,
    device: Any,
    apply_sigmoid: bool = False,
) -> dict[str, float]:
    import torch

    from utils.evaluation import evaluator

    model.eval()
    predict_list = []
    target_list = []

    with torch.no_grad():
        for batch in loader:
            source = batch[source_idx].to(device)
            target = batch[target_idx].to(device)
            neighbors = batch[neighbor_idx].to(device)

            outputs = model(source, neighbors)
            if apply_sigmoid:
                outputs = torch.sigmoid(outputs)

            predict_list.append(outputs)
            target_list.append(target)

    pearson_list, spearman_list, rmse_list = evaluator(predict_list, target_list)
    return {
        "pearson_mean": mean_or_nan(pearson_list),
        "pearson_std": std_or_nan(pearson_list),
        "spearman_mean": mean_or_nan(spearman_list),
        "spearman_std": std_or_nan(spearman_list),
        "rmse_mean": mean_or_nan(rmse_list),
        "rmse_std": std_or_nan(rmse_list),
    }


def _binary_auc_metrics(
    probabilities: Any,
    targets: Any,
    output_names: list[str],
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score

    metrics: dict[str, float] = {}
    auc_values = []

    for index, name in enumerate(output_names):
        prefix = name.lower().replace(" ", "_")
        truth = targets[:, index]
        probs = probabilities[:, index]

        try:
            auc_value = float(roc_auc_score(truth, probs))
            auc_values.append(auc_value)
        except ValueError:
            auc_value = float("nan")

        predictions = (probs > 0.5).astype(int)
        labels = np.unique(truth)
        if labels.size == 2:
            tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
            sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        else:
            sensitivity = float("nan")
            specificity = float("nan")

        metrics[f"{prefix}_auc"] = auc_value
        metrics[f"{prefix}_sensitivity"] = sensitivity
        metrics[f"{prefix}_specificity"] = specificity

    metrics["mean_auc"] = float(np.nanmean(auc_values)) if auc_values else float("nan")
    return metrics


def evaluate_binary(
    model: Any,
    loader: Any,
    source_idx: int,
    target_idx: int,
    neighbor_idx: int,
    device: Any,
    output_names: list[str],
) -> dict[str, float]:
    import torch

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            source = batch[source_idx].to(device)
            target = batch[target_idx].to(device)
            neighbors = batch[neighbor_idx].to(device)

            probs = torch.sigmoid(model(source, neighbors))
            all_probs.append(probs)
            all_targets.append(target)

    probabilities = torch.cat(all_probs, dim=0).cpu().numpy()
    targets = torch.cat(all_targets, dim=0).cpu().numpy()
    return _binary_auc_metrics(probabilities, targets, output_names)


def evaluate_binary_mean_auc(
    model: Any,
    loader: Any,
    source_idx: int,
    target_idx: int,
    neighbor_idx: int,
    device: Any,
) -> dict[str, float]:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            source = batch[source_idx].to(device)
            target = batch[target_idx].to(device)
            neighbors = batch[neighbor_idx].to(device)

            probs = torch.sigmoid(model(source, neighbors))
            all_probs.append(probs)
            all_targets.append(target)

    probabilities = torch.cat(all_probs, dim=0).cpu().numpy()
    targets = torch.cat(all_targets, dim=0).cpu().numpy()

    auc_values = []
    for index in range(probabilities.shape[1]):
        if len(set(targets[:, index])) <= 1:
            continue
        auc_values.append(float(roc_auc_score(targets[:, index], probabilities[:, index])))

    return {"mean_auroc": float(np.mean(auc_values)) if auc_values else float("nan")}


def build_optimizer(args: SimpleNamespace, model: Any) -> Any:
    import torch

    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: SimpleNamespace, optimizer: Any) -> Any | None:
    from torch.optim import lr_scheduler

    if getattr(args, "stepsize", 0) and args.stepsize > 0:
        return lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    return None


def default_output_dir(dataset_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("sweeps") / dataset_name / timestamp


def expand_runs(
    base_config: dict[str, Any],
    grid: dict[str, list[Any]],
    max_runs: int | None = None,
) -> list[dict[str, Any]]:
    if not grid:
        return [deepcopy(base_config)]

    keys = sorted(grid.keys())
    combos = []
    for values in itertools.product(*(grid[key] for key in keys)):
        config = deepcopy(base_config)
        config.update(dict(zip(keys, values)))
        combos.append(config)
        if max_runs is not None and len(combos) >= max_runs:
            break
    return combos


def print_run_plan(dataset_name: str, runs: list[dict[str, Any]], grid: dict[str, list[Any]]) -> None:
    print(f"Dataset: {dataset_name}")
    print(f"Planned runs: {len(runs)}")
    if grid:
        print("Grid:")
        for key in sorted(grid.keys()):
            print(f"  - {key}: {grid[key]}")
    if runs:
        preview = {key: runs[0][key] for key in sorted(runs[0].keys())}
        print("Example config:")
        print(json.dumps(serialise(preview), indent=2, sort_keys=True))


def dataset_spec(dataset_name: str) -> dict[str, Any]:
    if dataset_name == "sma":
        from datasets.data_manager_SMA import SMA
        from model.nicheTrans import NicheTrans
        from utils.utils_dataloader import sma_dataloader
        from utils.utils_training_SMA import train as train_epoch

        defaults = {
            "noise_rate": 0.2,
            "dropout_rate": 0.1,
            "use_moe_ffn": True,
            "num_experts": 1,
            "moe_gate_hidden_dim": 0,
            "moe_gate_type": "softmax",
            "ffn_mult": 2,
            "moe_router_temperature_enable": False,
            "moe_router_temperature_start": 1.0,
            "moe_router_temperature_mid": 0.7,
            "moe_router_temperature_end": 0.5,
            "moe_router_temperature_schedule": "step",
            "moe_balance_loss_enable": False,
            "moe_balance_loss_weight": 1e-3,
            "moe_balance_loss_type": "mse_uniform",
            "moe_router_entropy_penalty_enable": False,
            "moe_router_entropy_penalty_weight": 1e-3,
            "n_source": 3000,
            "n_target": 50,
            "img_size": 256,
            "workers": 4,
            "path_img": "/mnt/datadisk0/Processed_DATA/2023_nbt_SMA/Processed_data_used/patches",
            "rna_path": "/mnt/datadisk0/Processed_DATA/2023_nbt_SMA/Processed_data_used",
            "msi_path": "/mnt/datadisk0/Processed_DATA/2023_nbt_SMA/Processed_data_used",
            "max_epoch": 40,
            "stepsize": 20,
            "train_batch": 32,
            "test_batch": 32,
            "optimizer": "adam",
            "lr": 0.0003,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "seed": 1,
            "save_dir": "./log",
            "eval_step": 1,
            "gpu_devices": "0",
        }

        return {
            "defaults": defaults,
            "primary_metric_name": "pearson_mean",
            "higher_is_better": True,
            "build_dataset": lambda args: SMA(
                path_img=args.path_img,
                rna_path=args.rna_path,
                msi_path=args.msi_path,
                n_top_genes=args.n_source,
                n_top_targets=args.n_target,
            ),
            "build_loaders": lambda args, dataset: sma_dataloader(args, dataset),
            "build_model": lambda args, dataset: NicheTrans(
                source_length=dataset.rna_length,
                target_length=dataset.msi_length,
                noise_rate=args.noise_rate,
                dropout_rate=args.dropout_rate,
                use_moe_ffn=args.use_moe_ffn,
                num_experts=args.num_experts,
                moe_gate_hidden_dim=args.moe_gate_hidden_dim,
                moe_gate_type=args.moe_gate_type,
                ffn_mult=args.ffn_mult,
                moe_router_temperature_enable=args.moe_router_temperature_enable,
                moe_router_temperature_start=args.moe_router_temperature_start,
                moe_router_temperature_mid=args.moe_router_temperature_mid,
                moe_router_temperature_end=args.moe_router_temperature_end,
                moe_router_temperature_schedule=args.moe_router_temperature_schedule,
                moe_balance_loss_enable=args.moe_balance_loss_enable,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_balance_loss_type=args.moe_balance_loss_type,
                moe_router_entropy_penalty_enable=args.moe_router_entropy_penalty_enable,
                moe_router_entropy_penalty_weight=args.moe_router_entropy_penalty_weight,
            ),
            "build_criterion": lambda: __import__("torch").nn.MSELoss(),
            "train_one_epoch": lambda model, criterion, optimizer, trainloader, args, device: train_epoch(
                model,
                criterion,
                optimizer,
                trainloader,
                use_img=False,
                device=device,
            ),
            "evaluate": lambda model, loader, dataset, args, device: evaluate_regression(
                model,
                loader,
                source_idx=1,
                target_idx=2,
                neighbor_idx=3,
                device=device,
            ),
        }

    if dataset_name == "starmap_plus":
        from datasets.data_manager_STARmap_PLUS import AD_Mouse
        from model.nicheTrans import NicheTrans
        from utils.utils_dataloader import ad_mouse_dataloader
        from utils.utils_training_STARmap_PLUS import train as train_epoch

        defaults = {
            "noise_rate": 0.5,
            "dropout_rate": 0.25,
            "use_moe_ffn": True,
            "num_experts": 8,
            "moe_gate_hidden_dim": 512,
            "moe_gate_type": "softmax",
            "ffn_mult": 2,
            "moe_router_temperature_enable": False,
            "moe_router_temperature_start": 1.0,
            "moe_router_temperature_mid": 0.7,
            "moe_router_temperature_end": 0.5,
            "moe_router_temperature_schedule": "step",
            "moe_balance_loss_enable": False,
            "moe_balance_loss_weight": 1e-3,
            "moe_balance_loss_type": "mse_uniform",
            "moe_router_entropy_penalty_enable": False,
            "moe_router_entropy_penalty_weight": 1e-3,
            "n_top_genes": 2000,
            "workers": 4,
            "AD_adata_path": "/mnt/datadisk0/Processed_DATA/2023_nn_AD_mouse/AD_model_adata_protein",
            "Wild_type_adata_path": "/mnt/datadisk0/Processed_DATA/2023_nn_AD_mouse/wild_type_adata_protein",
            "max_epoch": 20,
            "stepsize": 20,
            "train_batch": 128,
            "test_batch": 32,
            "optimizer": "adam",
            "lr": 0.0001,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "seed": 1,
            "save_dir": "./log",
            "eval_step": 1,
            "gpu_devices": "0",
        }

        return {
            "defaults": defaults,
            "primary_metric_name": "mean_auc",
            "higher_is_better": True,
            "build_dataset": lambda args: AD_Mouse(
                AD_adata_path=args.AD_adata_path,
                Wild_type_adata_path=args.Wild_type_adata_path,
                n_top_genes=args.n_top_genes,
            ),
            "build_loaders": lambda args, dataset: ad_mouse_dataloader(args, dataset)[:2],
            "build_model": lambda args, dataset: NicheTrans(
                source_length=dataset.rna_length,
                target_length=dataset.target_length,
                noise_rate=args.noise_rate,
                dropout_rate=args.dropout_rate,
                use_moe_ffn=args.use_moe_ffn,
                num_experts=args.num_experts,
                moe_gate_hidden_dim=args.moe_gate_hidden_dim,
                moe_gate_type=args.moe_gate_type,
                ffn_mult=args.ffn_mult,
                moe_router_temperature_enable=args.moe_router_temperature_enable,
                moe_router_temperature_start=args.moe_router_temperature_start,
                moe_router_temperature_mid=args.moe_router_temperature_mid,
                moe_router_temperature_end=args.moe_router_temperature_end,
                moe_router_temperature_schedule=args.moe_router_temperature_schedule,
                moe_balance_loss_enable=args.moe_balance_loss_enable,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_balance_loss_type=args.moe_balance_loss_type,
                moe_router_entropy_penalty_enable=args.moe_router_entropy_penalty_enable,
                moe_router_entropy_penalty_weight=args.moe_router_entropy_penalty_weight,
            ),
            "build_criterion": lambda: __import__("torch").nn.BCELoss(),
            "train_one_epoch": lambda model, criterion, optimizer, trainloader, args, device: train_epoch(
                model,
                criterion,
                optimizer,
                trainloader,
                device=device,
            ),
            "evaluate": lambda model, loader, dataset, args, device: evaluate_binary(
                model,
                loader,
                source_idx=0,
                target_idx=1,
                neighbor_idx=3,
                device=device,
                output_names=["tau", "plaque"],
            ),
        }

    if dataset_name == "human_lymph_node":
        from datasets.data_manager_human_lymph_node import Lymph_node
        from model.nicheTrans import NicheTrans
        from utils.utils_dataloader import human_node_dataloader
        from utils.utils_training_human_lymph_node import train as train_epoch

        defaults = {
            "noise_rate": 0.5,
            "dropout_rate": 0.1,
            "use_moe_ffn": True,
            "num_experts": 4,
            "moe_gate_hidden_dim": 512,
            "moe_gate_type": "softmax",
            "ffn_mult": 2,
            "moe_router_temperature_enable": False,
            "moe_router_temperature_start": 1.0,
            "moe_router_temperature_mid": 0.7,
            "moe_router_temperature_end": 0.5,
            "moe_router_temperature_schedule": "step",
            "moe_balance_loss_enable": False,
            "moe_balance_loss_weight": 1e-3,
            "moe_balance_loss_type": "mse_uniform",
            "moe_router_entropy_penalty_enable": False,
            "moe_router_entropy_penalty_weight": 1e-3,
            "n_source": 3000,
            "workers": 4,
            "adata_path": "/mnt/datadisk0/Processed_DATA/2024_nm_human_lymph_nodes/",
            "max_epoch": 20,
            "stepsize": 10,
            "train_batch": 32,
            "test_batch": 32,
            "optimizer": "adam",
            "lr": 0.0003,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "seed": 1,
            "save_dir": "./log",
            "eval_step": 1,
            "gpu_devices": "0",
        }

        return {
            "defaults": defaults,
            "primary_metric_name": "pearson_mean",
            "higher_is_better": True,
            "build_dataset": lambda args: Lymph_node(
                adata_path=args.adata_path,
                n_top_genes=args.n_source,
            ),
            "build_loaders": lambda args, dataset: human_node_dataloader(args, dataset),
            "build_model": lambda args, dataset: NicheTrans(
                source_length=dataset.rna_length,
                target_length=dataset.protein_length,
                noise_rate=args.noise_rate,
                dropout_rate=args.dropout_rate,
                use_moe_ffn=args.use_moe_ffn,
                num_experts=args.num_experts,
                moe_gate_hidden_dim=args.moe_gate_hidden_dim,
                moe_gate_type=args.moe_gate_type,
                ffn_mult=args.ffn_mult,
                moe_router_temperature_enable=args.moe_router_temperature_enable,
                moe_router_temperature_start=args.moe_router_temperature_start,
                moe_router_temperature_mid=args.moe_router_temperature_mid,
                moe_router_temperature_end=args.moe_router_temperature_end,
                moe_router_temperature_schedule=args.moe_router_temperature_schedule,
                moe_balance_loss_enable=args.moe_balance_loss_enable,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_balance_loss_type=args.moe_balance_loss_type,
                moe_router_entropy_penalty_enable=args.moe_router_entropy_penalty_enable,
                moe_router_entropy_penalty_weight=args.moe_router_entropy_penalty_weight,
            ),
            "build_criterion": lambda: __import__("torch").nn.MSELoss(),
            "train_one_epoch": lambda model, criterion, optimizer, trainloader, args, device: train_epoch(
                model,
                criterion,
                optimizer,
                trainloader,
                device=device,
            ),
            "evaluate": lambda model, loader, dataset, args, device: evaluate_regression(
                model,
                loader,
                source_idx=0,
                target_idx=1,
                neighbor_idx=2,
                device=device,
            ),
        }

    if dataset_name == "breast_cancer":
        from datasets.data_manager_breast_cancer import Breast_cancer
        from model.nicheTrans import NicheTrans
        from utils.utils_dataloader import breast_cancer_dataloader
        from utils.utils_training_breast_cancer import train as train_epoch

        defaults = {
            "noise_rate": 0.2,
            "dropout_rate": 0.2,
            "use_moe_ffn": True,
            "num_experts": 1,
            "moe_gate_hidden_dim": 0,
            "moe_gate_type": "softmax",
            "ffn_mult": 2,
            "moe_router_temperature_enable": False,
            "moe_router_temperature_start": 1.0,
            "moe_router_temperature_mid": 0.7,
            "moe_router_temperature_end": 0.5,
            "moe_router_temperature_schedule": "step",
            "moe_balance_loss_enable": False,
            "moe_balance_loss_weight": 1e-3,
            "moe_balance_loss_type": "mse_uniform",
            "moe_router_entropy_penalty_enable": False,
            "moe_router_entropy_penalty_weight": 1e-3,
            "workers": 4,
            "adata_path": "/mnt/datadisk0/Processed_DATA/2023_nc_10x_breast_cancer/HBC_rep1_cell_nucleus_3channel_strength_mean.h5ad",
            "coordinate_path": "/mnt/datadisk0/Processed_DATA/2023_nc_10x_breast_cancer/cells.csv.gz",
            "ct_path": "/mnt/datadisk0/Processed_DATA/2023_nc_10x_breast_cancer/Cell_Barcode_Type_Matrices.xlsx",
            "max_epoch": 40,
            "stepsize": 20,
            "train_batch": 32,
            "test_batch": 32,
            "optimizer": "adam",
            "lr": 0.0003,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "seed": 1,
            "eval_step": 1,
            "gpu_devices": "0",
        }

        return {
            "defaults": defaults,
            "primary_metric_name": "pearson_mean",
            "higher_is_better": True,
            "build_dataset": lambda args: Breast_cancer(
                adata_path=args.adata_path,
                coordinate_path=args.coordinate_path,
                ct_path=args.ct_path,
            ),
            "build_loaders": lambda args, dataset: breast_cancer_dataloader(args, dataset),
            "build_model": lambda args, dataset: NicheTrans(
                source_length=dataset.rna_length,
                target_length=dataset.protein_length,
                noise_rate=args.noise_rate,
                dropout_rate=args.dropout_rate,
                use_moe_ffn=args.use_moe_ffn,
                num_experts=args.num_experts,
                moe_gate_hidden_dim=args.moe_gate_hidden_dim,
                moe_gate_type=args.moe_gate_type,
                ffn_mult=args.ffn_mult,
                moe_router_temperature_enable=args.moe_router_temperature_enable,
                moe_router_temperature_start=args.moe_router_temperature_start,
                moe_router_temperature_mid=args.moe_router_temperature_mid,
                moe_router_temperature_end=args.moe_router_temperature_end,
                moe_router_temperature_schedule=args.moe_router_temperature_schedule,
                moe_balance_loss_enable=args.moe_balance_loss_enable,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_balance_loss_type=args.moe_balance_loss_type,
                moe_router_entropy_penalty_enable=args.moe_router_entropy_penalty_enable,
                moe_router_entropy_penalty_weight=args.moe_router_entropy_penalty_weight,
            ),
            "build_criterion": lambda: __import__("torch").nn.MSELoss(),
            "train_one_epoch": lambda model, criterion, optimizer, trainloader, args, device: train_epoch(
                model,
                criterion,
                optimizer,
                trainloader,
                device=device,
            ),
            "evaluate": lambda model, loader, dataset, args, device: evaluate_regression(
                model,
                loader,
                source_idx=0,
                target_idx=1,
                neighbor_idx=2,
                device=device,
            ),
        }

    if dataset_name == "misar_atac2rna":
        from datasets.data_manager_MISAR_seq import ATAC_RNA_Seq
        from model.nicheTrans_hd import NicheTrans
        from utils.utils_dataloader import embryonic_mouse_brain
        from utils.utils_training_embryonic_mouse_brain import train_regression

        defaults = {
            "noise_rate": 0.2,
            "dropout_rate": 0.1,
            "use_moe_ffn": True,
            "num_experts": 1,
            "moe_gate_hidden_dim": 0,
            "moe_gate_type": "softmax",
            "ffn_mult": 2,
            "moe_router_temperature_enable": False,
            "moe_router_temperature_start": 1.0,
            "moe_router_temperature_mid": 0.7,
            "moe_router_temperature_end": 0.5,
            "moe_router_temperature_schedule": "step",
            "moe_balance_loss_enable": False,
            "moe_balance_loss_weight": 1e-3,
            "moe_balance_loss_type": "mse_uniform",
            "moe_router_entropy_penalty_enable": False,
            "moe_router_entropy_penalty_weight": 1e-3,
            "n_source": 3000,
            "workers": 4,
            "knn_smooth": True,
            "peak_threshold": 0.05,
            "hvg_gene": 1500,
            "adata_path": "/mnt/datadisk0/Processed_DATA/2023_nm_MISAR_seq",
            "max_epoch": 20,
            "stepsize": 10,
            "train_batch": 32,
            "test_batch": 32,
            "optimizer": "adam",
            "lr": 0.0003,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "seed": 1,
            "save_dir": "./log",
            "eval_step": 1,
            "gpu_devices": "0",
        }

        return {
            "defaults": defaults,
            "primary_metric_name": "pearson_mean",
            "higher_is_better": True,
            "build_dataset": lambda args: ATAC_RNA_Seq(
                peak_threshold=args.peak_threshold,
                hvg_gene=args.hvg_gene,
                adata_path=args.adata_path,
                RNA2ATAC=False,
                knn_smoothing=args.knn_smooth,
            ),
            "build_loaders": lambda args, dataset: embryonic_mouse_brain(args, dataset),
            "build_model": lambda args, dataset: NicheTrans(
                source_length=len(dataset.source_panel),
                target_length=len(dataset.target_panel),
                noise_rate=args.noise_rate,
                dropout_rate=args.dropout_rate,
                use_moe_ffn=args.use_moe_ffn,
                num_experts=args.num_experts,
                moe_gate_hidden_dim=args.moe_gate_hidden_dim,
                moe_gate_type=args.moe_gate_type,
                ffn_mult=args.ffn_mult,
                moe_router_temperature_enable=args.moe_router_temperature_enable,
                moe_router_temperature_start=args.moe_router_temperature_start,
                moe_router_temperature_mid=args.moe_router_temperature_mid,
                moe_router_temperature_end=args.moe_router_temperature_end,
                moe_router_temperature_schedule=args.moe_router_temperature_schedule,
                moe_balance_loss_enable=args.moe_balance_loss_enable,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_balance_loss_type=args.moe_balance_loss_type,
                moe_router_entropy_penalty_enable=args.moe_router_entropy_penalty_enable,
                moe_router_entropy_penalty_weight=args.moe_router_entropy_penalty_weight,
            ),
            "build_criterion": lambda: __import__("torch").nn.MSELoss(),
            "train_one_epoch": lambda model, criterion, optimizer, trainloader, args, device: train_regression(
                model,
                criterion,
                optimizer,
                trainloader,
                device=device,
            ),
            "evaluate": lambda model, loader, dataset, args, device: evaluate_regression(
                model,
                loader,
                source_idx=0,
                target_idx=1,
                neighbor_idx=2,
                device=device,
            ),
        }

    if dataset_name == "misar_rna2atac":
        from datasets.data_manager_MISAR_seq import ATAC_RNA_Seq
        from model.nicheTrans_hd import NicheTrans
        from utils.utils_dataloader import embryonic_mouse_brain
        from utils.utils_training_embryonic_mouse_brain import train_binary

        defaults = {
            "noise_rate": 0.2,
            "dropout_rate": 0.1,
            "use_moe_ffn": True,
            "num_experts": 1,
            "moe_gate_hidden_dim": 0,
            "moe_gate_type": "softmax",
            "ffn_mult": 2,
            "moe_router_temperature_enable": False,
            "moe_router_temperature_start": 1.0,
            "moe_router_temperature_mid": 0.7,
            "moe_router_temperature_end": 0.5,
            "moe_router_temperature_schedule": "step",
            "moe_balance_loss_enable": False,
            "moe_balance_loss_weight": 1e-3,
            "moe_balance_loss_type": "mse_uniform",
            "moe_router_entropy_penalty_enable": False,
            "moe_router_entropy_penalty_weight": 1e-3,
            "n_source": 3000,
            "workers": 4,
            "knn_smooth": True,
            "peak_threshold": 0.05,
            "hvg_gene": 1500,
            "adata_path": "/mnt/datadisk0/Processed_DATA/2023_nm_MISAR_seq",
            "max_epoch": 20,
            "stepsize": 10,
            "train_batch": 32,
            "test_batch": 32,
            "optimizer": "adam",
            "lr": 0.0003,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "seed": 1,
            "save_dir": "./log",
            "eval_step": 1,
            "gpu_devices": "0",
        }

        def evaluate(model: Any, loader: Any, dataset: Any, args: Any, device: Any) -> dict[str, float]:
            metrics = evaluate_regression(
                model,
                loader,
                source_idx=0,
                target_idx=1,
                neighbor_idx=2,
                device=device,
                apply_sigmoid=True,
            )
            metrics.update(
                evaluate_binary_mean_auc(
                    model,
                    loader,
                    source_idx=0,
                    target_idx=1,
                    neighbor_idx=2,
                    device=device,
                )
            )
            return metrics

        return {
            "defaults": defaults,
            "primary_metric_name": "pearson_mean",
            "higher_is_better": True,
            "build_dataset": lambda args: ATAC_RNA_Seq(
                peak_threshold=args.peak_threshold,
                hvg_gene=args.hvg_gene,
                adata_path=args.adata_path,
                RNA2ATAC=True,
                knn_smoothing=args.knn_smooth,
            ),
            "build_loaders": lambda args, dataset: embryonic_mouse_brain(args, dataset),
            "build_model": lambda args, dataset: NicheTrans(
                source_length=len(dataset.source_panel),
                target_length=len(dataset.target_panel),
                noise_rate=args.noise_rate,
                dropout_rate=args.dropout_rate,
                use_moe_ffn=args.use_moe_ffn,
                num_experts=args.num_experts,
                moe_gate_hidden_dim=args.moe_gate_hidden_dim,
                moe_gate_type=args.moe_gate_type,
                ffn_mult=args.ffn_mult,
                moe_router_temperature_enable=args.moe_router_temperature_enable,
                moe_router_temperature_start=args.moe_router_temperature_start,
                moe_router_temperature_mid=args.moe_router_temperature_mid,
                moe_router_temperature_end=args.moe_router_temperature_end,
                moe_router_temperature_schedule=args.moe_router_temperature_schedule,
                moe_balance_loss_enable=args.moe_balance_loss_enable,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_balance_loss_type=args.moe_balance_loss_type,
                moe_router_entropy_penalty_enable=args.moe_router_entropy_penalty_enable,
                moe_router_entropy_penalty_weight=args.moe_router_entropy_penalty_weight,
            ),
            "build_criterion": lambda: __import__("torch").nn.BCELoss(),
            "train_one_epoch": lambda model, criterion, optimizer, trainloader, args, device: train_binary(
                model,
                criterion,
                optimizer,
                trainloader,
                device=device,
            ),
            "evaluate": evaluate,
        }

    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. "
        "Choose from: sma, starmap_plus, human_lymph_node, breast_cancer, misar_atac2rna, misar_rna2atac."
    )


def should_update_best(
    score: float,
    best_score: float | None,
    higher_is_better: bool,
) -> bool:
    if math.isnan(score):
        return False
    if best_score is None or math.isnan(best_score):
        return True
    return score > best_score if higher_is_better else score < best_score


def run_single_experiment(
    run_index: int,
    num_runs: int,
    dataset_name: str,
    spec: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    fail_fast: bool,
) -> dict[str, Any]:
    run_name = f"run_{run_index:04d}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    summary_path = output_dir / "summary.jsonl"

    record: dict[str, Any] = {
        "run_name": run_name,
        "dataset": dataset_name,
        "status": "pending",
        **config,
    }

    stable_json_dump(config_path, {"dataset": dataset_name, "run_name": run_name, "config": config})
    print(f"\n[{run_index}/{num_runs}] Starting {run_name}")

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_devices", "0"))

        import torch

        from utils.utils import set_seed

        args = namespace_from_dict(config)
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = spec["build_dataset"](args)
        trainloader, testloader = spec["build_loaders"](args, dataset)
        model = spec["build_model"](args, dataset)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        criterion = spec["build_criterion"]()
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

        best_score = None
        best_epoch = None
        best_metrics: dict[str, Any] = {}
        last_metrics: dict[str, Any] = {}

        started_at = time.time()
        eval_step = getattr(args, "eval_step", 1)

        for epoch in range(1, args.max_epoch + 1):
            print(f"==> Epoch {epoch}/{args.max_epoch}")
            train_metrics = spec["train_one_epoch"](model, criterion, optimizer, trainloader, args, device)
            if scheduler is not None:
                scheduler.step()

            should_eval = eval_step == -1 and epoch == args.max_epoch
            should_eval = should_eval or (eval_step > 0 and (epoch % eval_step == 0 or epoch == args.max_epoch))

            if not should_eval:
                continue

            metrics = spec["evaluate"](model, testloader, dataset, args, device)
            score = float(metrics.get(spec["primary_metric_name"], float("nan")))
            train_history = {}
            if isinstance(train_metrics, dict):
                train_history = {f"train_{key}": value for key, value in train_metrics.items()}
            append_jsonl(
                history_path,
                {
                    "epoch": epoch,
                    "primary_metric_name": spec["primary_metric_name"],
                    "primary_metric": score,
                    **train_history,
                    **metrics,
                },
            )
            last_metrics = metrics

            if should_update_best(score, best_score, spec["higher_is_better"]):
                best_score = score
                best_epoch = epoch
                best_metrics = deepcopy(metrics)
                torch.save(torch_state_dict(model), run_dir / "best_model.pth")

        torch.save(torch_state_dict(model), run_dir / "last_model.pth")
        elapsed_seconds = round(time.time() - started_at, 2)

        if not best_metrics:
            best_metrics = deepcopy(last_metrics)
            best_score = float(best_metrics.get(spec["primary_metric_name"], float("nan")))
            best_epoch = args.max_epoch

        final_metrics = {
            "primary_metric_name": spec["primary_metric_name"],
            "primary_metric": best_score,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed_seconds,
            **ensure_metric_fields(best_metrics),
        }
        stable_json_dump(
            metrics_path,
            {
                "dataset": dataset_name,
                "run_name": run_name,
                "best_epoch": best_epoch,
                "primary_metric_name": spec["primary_metric_name"],
                "primary_metric": best_score,
                "elapsed_seconds": elapsed_seconds,
                "metrics": best_metrics,
            },
        )
        append_jsonl(
            summary_path,
            {
                "run_name": run_name,
                "dataset": dataset_name,
                "status": "success",
                **config,
                **final_metrics,
            },
        )

        record.update({"status": "success", **final_metrics})
        return record
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc)})
        stable_json_dump(metrics_path, {"dataset": dataset_name, "run_name": run_name, "status": "failed", "error": str(exc)})
        append_jsonl(summary_path, record)
        if fail_fast:
            raise
        print(f"Run {run_name} failed: {exc}", file=sys.stderr)
        return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NicheTrans hyper-parameter sweeps and record evaluation metrics.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "sma",
            "starmap_plus",
            "human_lymph_node",
            "breast_cancer",
            "misar_atac2rna",
            "misar_rna2atac",
        ],
        help="Training recipe to run.",
    )
    parser.add_argument("--set", action="append", default=[], help="Fixed override in KEY=VALUE format. Repeatable.")
    parser.add_argument("--grid", action="append", default=[], help="Grid override in KEY=V1,V2,... format. Repeatable.")
    parser.add_argument("--grid-file", type=str, default=None, help="Optional JSON file with {\"set\": {...}, \"grid\": {...}}.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where sweep outputs should be written. Defaults to ./sweeps/<dataset>/<timestamp>.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the number of generated combinations.")
    parser.add_argument("--dry-run", action="store_true", help="Only expand and print the planned runs without training.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the sweep immediately if one run fails.")
    return parser


def main() -> int:
    parser = build_parser()
    cli_args = parser.parse_args()

    file_config = load_json_config(cli_args.grid_file)
    file_set = {normalize_key(key): value for key, value in file_config.get("set", {}).items()}
    file_grid = {normalize_key(key): normalize_grid_values(value) for key, value in file_config.get("grid", {}).items()}

    cli_set = dict(parse_assignment(item) for item in cli_args.set)
    cli_grid = dict(parse_grid_assignment(item) for item in cli_args.grid)

    spec = dataset_spec(cli_args.dataset)
    base_config = deepcopy(spec["defaults"])

    known_keys = set(base_config.keys())
    override_keys = set(file_set) | set(file_grid) | set(cli_set) | set(cli_grid)
    unknown_keys = sorted(override_keys - known_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown parameter(s) for dataset '{cli_args.dataset}': {', '.join(unknown_keys)}"
        )

    base_config.update(file_set)
    base_config.update(cli_set)

    grid = deepcopy(file_grid)
    grid.update(cli_grid)

    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else default_output_dir(cli_args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = expand_runs(base_config, grid, max_runs=cli_args.max_runs)
    print_run_plan(cli_args.dataset, runs, grid)

    stable_json_dump(
        output_dir / "sweep_config.json",
        {
            "dataset": cli_args.dataset,
            "base_config": base_config,
            "grid": grid,
            "num_runs": len(runs),
        },
    )

    if cli_args.dry_run:
        return 0

    summary_rows = []
    config_keys = sorted(set(base_config.keys()) | set(grid.keys()))
    fieldnames = [
        "run_name",
        "dataset",
        "status",
        *config_keys,
        "primary_metric_name",
        "primary_metric",
        "best_epoch",
        "elapsed_seconds",
        *COMMON_METRIC_KEYS,
        "error",
    ]

    for index, config in enumerate(runs, start=1):
        row = run_single_experiment(
            run_index=index,
            num_runs=len(runs),
            dataset_name=cli_args.dataset,
            spec=spec,
            config=config,
            output_dir=output_dir,
            fail_fast=cli_args.fail_fast,
        )
        summary_rows.append(row)
        save_summary_row(output_dir / "summary.csv", row, fieldnames)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    print(f"\nFinished sweep: {success_count}/{len(summary_rows)} runs succeeded.")
    print(f"Outputs written to: {output_dir}")
    return 0 if success_count == len(summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
