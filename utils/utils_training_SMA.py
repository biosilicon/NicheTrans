import random
import torch

from utils.evaluation import evaluator
from utils.moe_training import (
    combine_task_and_moe_loss,
    finalize_metric_totals,
    prepare_moe_epoch,
    unpack_model_outputs,
    update_metric_totals,
)
from utils.utils import AverageMeter


def train(model, criterion, optimizer, trainloader, use_img=True, device=None, epoch=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prepare_moe_epoch(model, epoch=epoch)
    model.train()
    losses = AverageMeter()
    metric_totals = {}
    num_samples = 0

    for batch_idx, (imgs, source, target, source_neightbors, _, _) in enumerate(trainloader):
        source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)

        if random.random() > 0.7:
            mask = torch.ones((source_neightbors.size(0), 8, 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            source_neightbors = source_neightbors * mask

        if use_img:
            imgs = imgs.to(device)
            outputs = model(imgs, source, source_neightbors, return_moe_info=True)
        else:
            outputs = model(source, source_neightbors, return_moe_info=True)

        predictions, moe_info = unpack_model_outputs(outputs)
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


def test(model, testloader, use_img=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    predict_list, target_list = [], []
    coordinates = []

    with torch.no_grad():
        for _, (imgs, source, target, source_neightbors, _, samples) in enumerate(testloader):
            source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)

            if use_img:
                imgs = imgs.to(device)
                outputs = model(imgs, source, source_neightbors)
            else:
                outputs = model(source, source_neightbors)

            predict_list.append(outputs)
            target_list.append(target)
            coordinates += samples

    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)
    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    pearson_std, spearman_std, rmse_std = pearson_sample_list.std(), spearman_sample_list.std(), rmse_list.std()

    print(
        'Testing Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}'.format(
            pearson_mean,
            pearson_std,
            spearman_mean,
            spearman_std,
            rmse_mean,
            rmse_std,
        )
    )

    return pearson_mean
