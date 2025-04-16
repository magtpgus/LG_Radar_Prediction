from sklearn.metrics import mean_squared_error
import numpy as np

def lg_nrmse(gt, preds):
    all_nrmse = [mean_squared_error(gt[:, idx], preds[:, idx], squared=False) / np.mean(np.abs(gt[:, idx])) for idx in range(14)]
    return 1.2 * np.sum(all_nrmse[:7]) + 1.0 * np.sum(all_nrmse[7:])
