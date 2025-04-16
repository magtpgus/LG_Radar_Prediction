from data_loader import load_data, get_label
from feature_engineer import add_features
from model_wrappers import MyMultiOutputRegressor_LGBM, MyMultiOutputRegressor_XGB
from utils import lg_nrmse
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from config import *
import numpy as np
import pandas as pd
import os

train, test, threshold = load_data()
train['label'] = get_label(train, threshold)
train = train.drop(columns=DROP_X_EXTRA)
test = test.drop(columns=DROP_X_EXTRA)

train_X = train.drop(columns=['ID', 'label'] + [f'Y_{i:02}' for i in range(1, 15)])
train_y = train[[f'Y_{i:02}' for i in range(1, 15)]]
label_y = train['label']
test = test.drop(columns=['ID'])

train_means = train_X.mean()
train_X = add_features(train_X, train_means)
test = add_features(test, train_means)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_preds = []
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, label_y)):
    X_tr, X_val = train_X.iloc[tr_idx], train_X.iloc[val_idx]
    y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]

    cat_model = CatBoostRegressor(verbose=0, loss_function='MultiRMSE')
    cat_model.fit(X_tr, y_tr)
    pred = cat_model.predict(X_val)
    score = lg_nrmse(y_val.values, pred)
    print(f"[Fold {fold+1}] CatBoost nrmse: {score:.4f}")
    fold_preds.append(cat_model.predict(test))
    fold_scores.append(score)

print(f"\nAvg nrmse: {np.mean(fold_scores):.4f}")

submission = pd.read_csv('./sample_submission.csv')
final_pred = np.mean(fold_preds, axis=0)
for i, col in enumerate(submission.columns):
    if col != 'ID':
        submission[col] = final_pred[:, i-1]
submission.to_csv('./final_pred.csv', index=False)
