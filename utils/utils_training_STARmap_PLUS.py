import numpy as np
import random
import torch

from sklearn.metrics import confusion_matrix, roc_auc_score

from utils.evaluation import evaluator
from utils.moe_training import (
    combine_task_and_moe_loss,
    finalize_metric_totals,
    prepare_moe_epoch,
    unpack_model_outputs,
    update_metric_totals,
)
from utils.utils import AverageMeter


def train(model, criterion, optimizer, trainloader, ct_information=False, device=None, epoch=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prepare_moe_epoch(model, epoch=epoch)
    model.train()
    losses = AverageMeter()
    metric_totals = {}
    num_samples = 0

    for batch_idx, (rna, protein, cell, rna_neighbors, cell_neighbor, _) in enumerate(trainloader):
        rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)
        cell, cell_neighbor = cell.to(device), cell_neighbor.to(device)

        if random.random() > 0.7:
            mask = torch.ones((rna_neighbors.size(0), rna_neighbors.size(1), 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            rna_neighbors = rna_neighbors * mask
            cell_neighbor = cell_neighbor * mask

        cell_inf = torch.cat([cell[:, None, :], cell_neighbor], dim=1)
        source, target, source_neightbors = rna, protein, rna_neighbors

        if ct_information:
            outputs = model(source, source_neightbors, cell_inf, return_moe_info=True)
        else:
            outputs = model(source, source_neightbors, return_moe_info=True)

        predictions, moe_info = unpack_model_outputs(outputs)
        predictions = torch.sigmoid(predictions)
        task_loss = criterion(predictions, target)
        loss, batch_metrics = combine_task_and_moe_loss(task_loss, moe_info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = source.size(0)
        losses.update(loss.item(), batch_size)
        update_metric_totals(metric_totals, batch_metrics, batch_size)
        num_samples += batch_size

        if (batch_idx + 1) == len(trainloader):
            summary = finalize_metric_totals(metric_totals, num_samples)
            print(
                "Batch {}/{}\t Loss {:.6f} ({:.6f}) | Task {:.6f} | Aux {:.6f} | Tau {:.4f} | Bal {:.6f} | Ent {:.6f} | Margin {:.6f} | Cos {:.6f}".format(
                    batch_idx + 1,
                    len(trainloader),
                    losses.val,
                    losses.avg,
                    summary.get("task_loss", 0.0),
                    summary.get("moe_aux_loss", 0.0),
                    summary.get("router_temperature", 1.0),
                    summary.get("balance_loss", 0.0),
                    summary.get("router_entropy_penalty", 0.0),
                    summary.get("mean_gate_margin", 0.0),
                    summary.get("expert_output_cosine_mean", 0.0),
                )
            )

    return finalize_metric_totals(metric_totals, num_samples)


def test(model, testloader, ct_information=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predict_list, target_list = [], []

    with torch.no_grad():
        for _, (source, target, cell, source_neightbors, cell_neighbor, _) in enumerate(testloader):
            source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)
            cell_inf = torch.cat([cell[:, None, :], cell_neighbor], dim=1).to(device)

            if ct_information:
                outputs = model(source, source_neightbors, cell_inf)
            else:
                outputs = model(source, source_neightbors)
            outputs = torch.sigmoid(outputs)

            predict_list.append(outputs)
            target_list.append(target)

    predict_list = torch.cat(predict_list, dim=0).cpu().numpy()
    target_list = torch.cat(target_list, dim=0).cpu().numpy()

    auc_tau = roc_auc_score(target_list[:, 0], predict_list[:, 0])
    tn, fp, fn, tp = confusion_matrix(target_list[:, 0], (predict_list[:, 0] > 0.5).astype(int)).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"tau AUC: {auc_tau}, tau sensitivity {sensitivity}, tay specificity {specificity}")

    auc_plaque = roc_auc_score(target_list[:, 1], predict_list[:, 1])
    tn, fp, fn, tp = confusion_matrix(target_list[:, 1], (predict_list[:, 1] > 0.5).astype(int)).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Aå°¾ AUC: {auc_plaque}, Aå°¾ sensitivity {sensitivity}, Aå°¾ specificity {specificity}")

    return None
