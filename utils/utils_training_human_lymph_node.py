import numpy as np

import random
import torch

from utils.evaluation import evaluator
from utils.graph_meta import get_batch_graph_meta
from utils.utils import AverageMeter


def train(model, criterion, optimizer, trainloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    losses = AverageMeter()

    for batch_idx, (rna, protein, rna_neighbors, samples) in enumerate(trainloader):

        rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)
        neighbor_mask = None

        ############
        if random.random() > 0.7:
            mask = torch.ones((rna_neighbors.size(0), 8, 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            rna_neighbors = rna_neighbors * mask
            neighbor_mask = mask
        ############

        source, target, source_neightbors = rna, protein, rna_neighbors
        graph_meta = get_batch_graph_meta(trainloader.dataset, samples, device=device, neighbor_keep_mask=neighbor_mask)
        outputs = model(source, source_neightbors, graph_meta=graph_meta)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, source.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, testloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predict_list, target_list = [], []


    with torch.no_grad():
        for _, (source, target, source_neightbors, samples) in enumerate(testloader):

            source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)
            graph_meta = get_batch_graph_meta(testloader.dataset, samples, device=device)
            outputs = model(source, source_neightbors, graph_meta=graph_meta)

            predict_list.append(outputs)
            target_list.append(target)



    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)
    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    pearson_std, spearman_std, rmse_std = pearson_sample_list.std(), spearman_sample_list.std(), rmse_list.std()

    print('Testing Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}'
                                            .format(pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std))

    return pearson_mean
