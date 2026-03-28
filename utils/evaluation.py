import torch
import numpy as np
import warnings
from scipy.stats import rankdata


def draw_dot_plots(predict_list, target_list, pearson_sample_list, norm_rmse_list, panel, training=True):
    import matplotlib.pyplot as plt

    for i in range(predict_list.shape[1]):

        predict = predict_list[:, i]
        targets = target_list[:, i]

        max_val = max(max(predict), max(targets))
        lim = (0, max_val)
        plt.plot(lim, lim, color='red', linestyle='--')

        plt.figure(figsize=(8, 6))
        plt.scatter(predict, targets, color='blue', alpha=0.5)
        plt.plot(lim, lim, color='red', linestyle='--', linewidth=2)

        plt.xlim(lim)
        plt.ylim(lim)

        plt.xlabel('Predicted Number')
        plt.ylabel('Ground Truth')
        plt.title('Predicted vs. Ground Truth: Pearson {} and norm rmse {} of Panel {}'.format(str(pearson_sample_list[i])[0:5], str(norm_rmse_list[i])[0:5], panel[i]))
        plt.grid(True)
        if training==True:
            plt.savefig('./plots/training_{}_{}.jpg'.format(panel[i], str(i)))
        else:
            plt.savefig('./plots/testing_{}_{}.jpg'.format(panel[i], str(i)))
        plt.close()


def evaluator(predict_list, target_list):

    if isinstance(predict_list, list):
        predict_list = torch.cat(predict_list, dim=0).cpu().detach().numpy()
    if isinstance(target_list, list):
        target_list = torch.cat(target_list, dim=0).cpu().detach().numpy()

    num_targets = target_list.shape[1]

    # Vectorized Pearson correlation across all targets
    pred_centered = predict_list - predict_list.mean(axis=0, keepdims=True)
    targ_centered = target_list - target_list.mean(axis=0, keepdims=True)
    cov = (pred_centered * targ_centered).sum(axis=0)
    pred_std = np.sqrt((pred_centered ** 2).sum(axis=0))
    targ_std = np.sqrt((targ_centered ** 2).sum(axis=0))
    denom = pred_std * targ_std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pearson_all = np.where(denom > 0, cov / denom, np.nan)

    # Vectorized Spearman correlation (Pearson on ranks)
    pred_ranks = np.apply_along_axis(rankdata, 0, predict_list)
    targ_ranks = np.apply_along_axis(rankdata, 0, target_list)
    pred_r_centered = pred_ranks - pred_ranks.mean(axis=0, keepdims=True)
    targ_r_centered = targ_ranks - targ_ranks.mean(axis=0, keepdims=True)
    cov_r = (pred_r_centered * targ_r_centered).sum(axis=0)
    pred_r_std = np.sqrt((pred_r_centered ** 2).sum(axis=0))
    targ_r_std = np.sqrt((targ_r_centered ** 2).sum(axis=0))
    denom_r = pred_r_std * targ_r_std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spearman_all = np.where(denom_r > 0, cov_r / denom_r, np.nan)

    # Vectorized RMSE
    rmse_all = np.sqrt(np.mean((predict_list - target_list) ** 2, axis=0))

    # Filter out NaN values (matching original behavior: skip targets with NaN correlations)
    valid_mask = ~(np.isnan(pearson_all) | np.isnan(spearman_all))
    pearson_sample_list = pearson_all[valid_mask]
    spearman_sample_list = spearman_all[valid_mask]
    rmse_list = rmse_all[valid_mask]

    return pearson_sample_list, spearman_sample_list, rmse_list
