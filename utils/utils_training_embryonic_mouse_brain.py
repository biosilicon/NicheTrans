import numpy as np

import random
import torch

from utils.evaluation import evaluator
from utils.utils import AverageMeter
from sklearn.metrics import roc_auc_score


def train_regression(model, criterion, optimizer, trainloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    losses = AverageMeter()

    for batch_idx, (source, target, source_neighbors, spot_type_ids, _) in enumerate(trainloader):

        source = source.to(device)
        target = target.to(device)
        source_neighbors = source_neighbors.to(device)
        spot_type_ids = spot_type_ids.to(device)

        ############
        if random.random() > 0.7:
            mask = torch.ones((source_neighbors.size(0), 8, 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            source_neighbors = source_neighbors * mask
        ############

        outputs = model(source, source_neighbors, spot_type=spot_type_ids)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, source.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def train_binary(model, criterion, optimizer, trainloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    losses = AverageMeter()

    for batch_idx, (source, target, source_neighbors, spot_type_ids, _) in enumerate(trainloader):

        source = source.to(device)
        target = target.to(device)
        source_neighbors = source_neighbors.to(device)
        spot_type_ids = spot_type_ids.to(device)

        ############
        if random.random() > 0.7:
            mask = torch.ones((source_neighbors.size(0), 8, 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            source_neighbors = source_neighbors * mask
        ############

        outputs = model(source, source_neighbors, spot_type=spot_type_ids)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, source.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))



def test_regression(model, testloader, if_sigmoid=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predict_list, target_list = [], []


    with torch.no_grad():
        for _, (source, target, source_neightbors, spot_type_ids, _) in enumerate(testloader):

            source = source.to(device)
            target = target.to(device)
            source_neightbors = source_neightbors.to(device)
            spot_type_ids = spot_type_ids.to(device)
            outputs = model(source, source_neightbors, spot_type=spot_type_ids)

            if if_sigmoid:
                outputs = torch.sigmoid(outputs)

            predict_list.append(outputs)
            target_list.append(target)

    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)

    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    print('Testing Set: pearson correlation {:.4f}; spearman correlation {:.4f}; rmse {:.4f}'.format(pearson_mean, spearman_mean, rmse_mean))

    return pearson_mean


def test_binary(model, testloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predict_list, target_list = [], []


    with torch.no_grad():
        for _, (source, target, source_neightbors, spot_type_ids, _) in enumerate(testloader):

            source = source.to(device)
            target = target.to(device)
            source_neightbors = source_neightbors.to(device)
            spot_type_ids = spot_type_ids.to(device)
            outputs = model(source, source_neightbors, spot_type=spot_type_ids)
            outputs = torch.sigmoid(outputs)

            predict_list.append(outputs)
            target_list.append(target)

        auroc_list = []
        predict_list = torch.cat(predict_list, dim=0).cpu().numpy()
        target_list = torch.cat(target_list, dim=0).cpu().numpy()

        for i in range(predict_list.shape[1]):
            if len(set(target_list[:, i])) == 1:
                continue
            else:
                auroc_list.append(roc_auc_score(target_list[:, i], predict_list[:, i]))

        mean_auroc = np.mean(auroc_list)
        print(f'Testing Set: mean auroc {mean_auroc}')

        return mean_auroc
