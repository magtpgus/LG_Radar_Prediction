import numpy as np

def get_values(data_df, start_val, end_val, get_std=False):
    diff = []
    std = []
    for i in range(data_df.shape[0]):
        vals = [data_df[f"X_{s}"].iloc[i] for s in range(start_val, end_val+1)]
        diff.append(max(vals) - min(vals))
        if get_std:
            std.append(np.std(vals))
    return diff, std

def get_sum_values(data_df, val_list):
    return [sum([data_df[f"X_{s}"].iloc[i] for s in val_list]) for i in range(data_df.shape[0])]

def add_features(df, train_df_means):
    df["X_03/X_07"] = df["X_03"] / df["X_07"]
    diff, _ = get_values(df, 41, 44)
    df["X_41~44-diff"] = diff
    df["X_1~6_push-sum"] = get_sum_values(df, [1, 2, 5, 6])
    df["X_7~9_area-sum"] = get_sum_values(df, [7, 8, 9])
    df["X_03/X_19~22"] = df["X_03"] / (df["X_19"] + df["X_20"] + df["X_21"] + df["X_22"])
    df["X_12/X_24~25"] = df["X_12"] / ((df["X_24"] + df["X_25"]) / 2)
    df["49_7_19_3_8"] = (
        df["X_49"] / train_df_means["X_49"] +
        df["X_07"] / train_df_means["X_07"] +
        df["X_19"] / train_df_means["X_19"] +
        df["X_03"] / train_df_means["X_03"]
    )
    return df
