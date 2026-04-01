import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.evaluation import evaluator
from utils.utils import AverageMeter


def _resolve_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_mask(batch, split):
    mask_name = f"{split}_mask"
    if not hasattr(batch, mask_name):
        raise AttributeError(f"Batch is missing required mask '{mask_name}'.")
    return getattr(batch, mask_name)


def train_epoch(model, criterion, optimizer, trainloader, split="train", device=None):
    device = _resolve_device(device)
    model.train()
    losses = AverageMeter()

    for batch_idx, batch in enumerate(trainloader):
        batch = batch.to(device)
        mask = _get_mask(batch, split)
        if int(mask.sum()) == 0:
            continue

        pred = model(batch)
        loss = criterion(pred[mask], batch.y[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), int(mask.sum()))

        if (batch_idx + 1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    return losses.avg


def evaluate(model, dataloader, criterion=None, split="val", task_type="regression", device=None):
    device = _resolve_device(device)
    model.eval()

    losses = AverageMeter()
    predict_list, target_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mask = _get_mask(batch, split)
            if int(mask.sum()) == 0:
                continue

            pred = model(batch)
            target = batch.y

            if criterion is not None:
                loss = criterion(pred[mask], target[mask])
                losses.update(loss.item(), int(mask.sum()))

            predict_list.append(pred[mask].detach().cpu())
            target_list.append(target[mask].detach().cpu())

    if not predict_list:
        return {"loss": float("nan")}

    if task_type == "binary":
        predict = torch.sigmoid(torch.cat(predict_list, dim=0)).numpy()
        target = torch.cat(target_list, dim=0).numpy()

        auroc_list = []
        for idx in range(predict.shape[1]):
            if len(set(target[:, idx].astype(int).tolist())) < 2:
                continue
            auroc_list.append(roc_auc_score(target[:, idx], predict[:, idx]))

        mean_auroc = float(np.mean(auroc_list)) if auroc_list else float("nan")
        result = {"loss": losses.avg, "mean_auroc": mean_auroc}
        print(f"{split.title()} Set: mean auroc {mean_auroc}")
        return result

    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)
    pearson_mean = float(pearson_sample_list.mean()) if pearson_sample_list.size else float("nan")
    spearman_mean = float(spearman_sample_list.mean()) if spearman_sample_list.size else float("nan")
    rmse_mean = float(rmse_list.mean()) if rmse_list.size else float("nan")
    pearson_std = float(pearson_sample_list.std()) if pearson_sample_list.size else float("nan")
    spearman_std = float(spearman_sample_list.std()) if spearman_sample_list.size else float("nan")
    rmse_std = float(rmse_list.std()) if rmse_list.size else float("nan")

    print(
        "{} Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}".format(
            split.title(),
            pearson_mean,
            pearson_std,
            spearman_mean,
            spearman_std,
            rmse_mean,
            rmse_std,
        )
    )
    return {
        "loss": losses.avg,
        "pearson_mean": pearson_mean,
        "spearman_mean": spearman_mean,
        "rmse_mean": rmse_mean,
    }
