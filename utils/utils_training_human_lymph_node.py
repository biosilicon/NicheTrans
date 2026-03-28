import numpy as np

import random
import torch

from utils.evaluation import evaluator
from utils.utils import AverageMeter


def train(model, criterion, optimizer, trainloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model.train()
    losses = AverageMeter()

    for batch_idx, (rna, protein, rna_neighbors, _) in enumerate(trainloader):

        rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)

        ############
        if random.random() > 0.7:
            mask = torch.bernoulli(torch.full(
                (rna_neighbors.size(0), 8, 1), 0.5, device=rna_neighbors.device))
            rna_neighbors = rna_neighbors * mask
        ############

        source, target, source_neightbors = rna, protein, rna_neighbors

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(source, source_neightbors)
            loss = criterion(outputs, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.data, source.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, testloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_amp = device.type == 'cuda'
    model.eval()

    predict_list, target_list = [], []

    with torch.no_grad():
        for _, (source, target, source_neightbors, _) in enumerate(testloader):

            source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(source, source_neightbors)

            predict_list.append(outputs)
            target_list.append(target)


    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)
    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    pearson_std, spearman_std, rmse_std = pearson_sample_list.std(), spearman_sample_list.std(), rmse_list.std()

    print('Testing Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}'
                                            .format(pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std))

    return pearson_mean
