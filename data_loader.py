import pandas as pd
from config import *

def load_data():
    data_train = pd.read_csv('./train.csv')
    data_test = pd.read_csv('./test.csv')
    data_threshold = pd.read_csv('./y_feature_spec_info.csv')
    return data_train, data_test, data_threshold

def get_label(data_df, data_threshold):
    label = []
    for i in range(data_df.shape[0]):
        is_anomaly = False
        for idx in range(len(data_threshold)):
            col = data_threshold["Feature"].iloc[idx]
            min_val = data_threshold["최소"].iloc[idx]
            max_val = data_threshold["최대"].iloc[idx]
            val = data_df[col].iloc[i]
            if val < min_val or val > max_val:
                is_anomaly = True
                break
        label.append(1 if is_anomaly else 0)
    return label
