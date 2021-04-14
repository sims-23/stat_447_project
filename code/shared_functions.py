import pandas as pd
import numpy as np
import re


# Given the row from a dataframe, get the order of values if the row was sorted in the increasing order
def get_order(row):
    list_row = list(row)
    # row.values.tolist()
    return np.argsort(list_row)


# Constructs Prediction Intervals and saves it into csv
def category_pred_interval(prob_matrix, labels, prob_value, y_true):
    n_cases = prob_matrix.shape[0]
    pred_list: list = [1] * n_cases
    for i in range(0, n_cases):
        p = prob_matrix[i]
        ip = get_order(p)
        p_ordered = list(p[ip])
        labels_ordered = [labels[i] for i in ip[::-1]]
        g = list(np.cumsum([0] + p_ordered))[::-1]
        k = np.min(np.where(np.array(g) <= (1-prob_value))) - 1
        pred_labels = list(labels_ordered[:k+1])
        pred_list[i] = '.'.join([str(p) for p in pred_labels])
    true_list = list(y_true)
    true_list = [str(p) for p in true_list]
    df_pred_interval = pd.DataFrame(list(zip(true_list, pred_list)), columns=['True', 'Predicted'])
    tab = df_pred_interval.groupby(['True', 'Predicted']).size().unstack().fillna(value=0)
    return tab


def coverage(table):
    n_class = table.shape[0]
    n_subset = table.shape[1]
    labels = table.index.values.tolist()
    subset_labels = table.columns.values.tolist()
    row_freq = np.sum(table, axis=1)
    cov = np.zeros(n_class)
    avg_len = np.zeros(n_class)
    for i in range(n_class):
        for j in range(n_subset):
            n_char = len(re.split("\.", subset_labels[j]))
            interval_size = n_char
            is_covered = [bool(re.search(labels[i], subset_labels[j]))]
            freq = table.iloc[i, j]
            cov[i] = cov[i] + freq * int(np.any(is_covered))
            avg_len[i] = avg_len[i] + freq * interval_size
    miss = row_freq-cov
    avg_len = avg_len/row_freq

    return {'avg_len': avg_len, 'miss': miss, 'miss_rate': miss/row_freq, 'cov_rate': cov/row_freq}
