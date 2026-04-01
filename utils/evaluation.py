import torch
import scipy
import numpy as np


def evaluator(predict_list, target_list):

    if isinstance(predict_list, list):
        predict_list = torch.cat(predict_list, dim=0).cpu().detach().numpy()
    if isinstance(target_list, list):
        target_list = torch.cat(target_list, dim=0).cpu().detach().numpy()

    pearson_sample_list, spearman_sample_list, rmse_list = [], [], []
    num_targets = target_list.shape[1]

    for i in range(num_targets):
        pearson_corr, _ = scipy.stats.pearsonr(predict_list[:, i], target_list[:, i])
        spearman_corr, _ = scipy.stats.spearmanr(predict_list[:, i], target_list[:, i])
        rmse =  np.sqrt(np.mean( (predict_list[:, i] - target_list[:, i])**2 ))

        if np.isnan(pearson_corr): continue
        else: pearson_sample_list.append(pearson_corr)

        if np.isnan(spearman_corr): continue
        else: spearman_sample_list.append(spearman_corr)
        
        rmse_list.append(rmse)

    pearson_sample_list = np.array(pearson_sample_list)
    spearman_sample_list = np.array(spearman_sample_list) 
    rmse_list = np.array(rmse_list)

    return pearson_sample_list, spearman_sample_list, rmse_list
