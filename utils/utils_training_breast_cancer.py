import numpy as np

import random
import torch

from utils.utils import AverageMeter
from utils.evaluation import evaluator
from utils.graph_meta import get_batch_graph_meta


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
            mask = torch.ones((rna_neighbors.size(0), rna_neighbors.size(1), 1))
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
        for _, (rna, protein, rna_neighbors, samples) in enumerate(testloader):
            rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)
            source, target, source_neightbors = rna, protein, rna_neighbors
            graph_meta = get_batch_graph_meta(testloader.dataset, samples, device=device)

            outputs = model(source, source_neightbors, graph_meta=graph_meta)

            predict_list.append(outputs)
            target_list.append(target)


    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)

    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    print('Testing Set: pearson correlation {:.4f}; spearman correlation {:.4f}; rmse {:.4f}'.format(pearson_mean, spearman_mean, rmse_mean))

    return pearson_mean
