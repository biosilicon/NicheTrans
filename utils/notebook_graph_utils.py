from __future__ import annotations

from copy import deepcopy

import torch

from model.whole_slice_graph_transformer import WholeSliceGraphTransformer


def resolve_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset_dims(dataset):
    if hasattr(dataset, "rna_length"):
        source_dim = int(dataset.rna_length)
    else:
        source_dim = int(len(dataset.source_panel))

    if hasattr(dataset, "target_length"):
        target_dim = int(dataset.target_length)
    elif hasattr(dataset, "protein_length"):
        target_dim = int(dataset.protein_length)
    elif hasattr(dataset, "msi_length"):
        target_dim = int(dataset.msi_length)
    else:
        target_dim = int(len(dataset.target_panel))

    return source_dim, target_dim


def build_graph_model(args, dataset, use_cell_type=False):
    source_dim, target_dim = dataset_dims(dataset)
    num_cell_types = int(getattr(dataset, "num_cell_types", 0)) if use_cell_type else 0

    model = WholeSliceGraphTransformer(
        source_dim=source_dim,
        target_dim=target_dim,
        hidden_dim=args.graph_hidden_dim,
        num_layers=args.graph_num_layers,
        heads=args.graph_heads,
        dropout_rate=args.dropout_rate,
        noise_rate=args.noise_rate,
        use_cell_type=bool(use_cell_type),
        num_cell_types=num_cell_types,
    )
    return model


def state_dict_for_saving(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_graph_checkpoint(model, checkpoint_path, device=None):
    device = resolve_device(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def graph_mask(graph, split="test"):
    return getattr(graph, f"{split}_mask")


def collect_predictions(model, graphs, split="test", device=None, apply_sigmoid=False, clip_min=None):
    device = resolve_device(device)
    model.eval()

    predict_list = []
    target_list = []
    node_ids = []

    with torch.no_grad():
        for graph in graphs:
            batch = deepcopy(graph).to(device)
            mask = graph_mask(batch, split=split)
            pred = model(batch)

            if apply_sigmoid:
                pred = torch.sigmoid(pred)
            if clip_min is not None:
                pred = torch.clamp(pred, min=clip_min)

            predict_list.append(pred[mask].detach().cpu())
            target_list.append(batch.y[mask].detach().cpu())

            graph_node_ids = getattr(graph, "node_ids", None)
            if graph_node_ids is not None:
                mask_list = mask.detach().cpu().tolist()
                node_ids.extend([node_id for node_id, keep in zip(graph_node_ids, mask_list) if keep])

    predict = torch.cat(predict_list, dim=0)
    target = torch.cat(target_list, dim=0)
    return predict, target, node_ids


def predict_single_graph(model, graph, device=None, apply_sigmoid=False):
    device = resolve_device(device)
    batch = deepcopy(graph).to(device)

    with torch.no_grad():
        pred = model(batch)
        if apply_sigmoid:
            pred = torch.sigmoid(pred)

    return pred.detach().cpu(), batch.y.detach().cpu(), graph_mask(batch, split="test").detach().cpu()


def feature_gradients(model, graph, target_index, split="test", device=None, apply_sigmoid=False, selection_mask=None):
    device = resolve_device(device)
    batch = deepcopy(graph).to(device)
    batch.x = batch.x.detach().clone().requires_grad_(True)

    model.eval()
    model.zero_grad(set_to_none=True)

    outputs = model(batch)
    if apply_sigmoid:
        outputs = torch.sigmoid(outputs)

    mask = graph_mask(batch, split=split)
    if selection_mask is not None:
        mask = mask & selection_mask.to(device)

    if int(mask.sum()) == 0:
        raise ValueError("No nodes selected for attribution.")

    score = outputs[mask, target_index].sum()
    score.backward()

    gradients = batch.x.grad.detach().cpu()
    return gradients, outputs.detach().cpu(), mask.detach().cpu()


def cell_type_gradients(model, graph, target_index, split="test", device=None, apply_sigmoid=False, selection_mask=None):
    if model.cell_type_embedding is None:
        raise ValueError("Model was created without cell-type embeddings.")

    device = resolve_device(device)
    batch = deepcopy(graph).to(device)

    model.eval()
    model.zero_grad(set_to_none=True)

    outputs = model(batch)
    if apply_sigmoid:
        outputs = torch.sigmoid(outputs)

    mask = graph_mask(batch, split=split)
    if selection_mask is not None:
        mask = mask & selection_mask.to(device)

    if int(mask.sum()) == 0:
        raise ValueError("No nodes selected for attribution.")

    score = outputs[mask, target_index].sum()
    score.backward()

    gradients = model.cell_type_embedding.weight.grad.detach().cpu()
    return gradients, outputs.detach().cpu(), mask.detach().cpu()
