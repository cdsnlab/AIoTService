import os

import numpy as np
from sklearn.metrics import confusion_matrix


def exponentialDecay(N, decay_weight):
    tau = 1
    tmax = 4
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    return y/float(decay_weight)
    # return y/10.

def metrics(true_y, pred_y):
    metrics = {}
    cm = confusion_matrix(true_y, pred_y)
    support = np.unique(true_y, return_counts = True)[1]

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    metrics['TPR'] = TP / (TP + FN) # Sensitivity, hit rate, recall, or true positive rate
    metrics['TNR'] = TN / (TN + FP) # Specificity or true negative rate
    metrics['PPV'] = TP / (TP + FP) # Precision or positive predictive value
    metrics['NPV'] = TN / (TN + FN) # Negative predictive value
    metrics['FPR'] = FP / (FP + TN) # Fall out or false positive rate
    metrics['FNR'] = FN / (TP + FN) # False negative rate
    metrics['FDR'] = FP / (TP + FP) # False discovery rate
    metrics['ACC'] = (TP + TN) / (TP + FP + FN + TN) # Overall accuracy
    
    metrics['micro_TPR'] = (metrics['TPR'] * (support / support.sum())).sum()
    metrics['micro_TNR'] = (metrics['TNR'] * (support / support.sum())).sum()
    metrics['micro_PPV'] = (metrics['PPV'] * (support / support.sum())).sum()
    metrics['micro_NPV'] = (metrics['NPV'] * (support / support.sum())).sum()
    metrics['micro_FPR'] = (metrics['FPR'] * (support / support.sum())).sum()
    metrics['micro_FNR'] = (metrics['FNR'] * (support / support.sum())).sum()
    metrics['micro_FDR'] = (metrics['FDR'] * (support / support.sum())).sum()
    metrics['micro_ACC'] = (metrics['ACC'] * (support / support.sum())).sum()
    
    metrics['macro_TPR'] = metrics['TPR'].mean()
    metrics['macro_TNR'] = metrics['TNR'].mean()
    metrics['macro_PPV'] = metrics['PPV'].mean()
    metrics['macro_NPV'] = metrics['NPV'].mean()
    metrics['macro_FPR'] = metrics['FPR'].mean()
    metrics['macro_FNR'] = metrics['FNR'].mean()
    metrics['macro_FDR'] = metrics['FDR'].mean()
    metrics['macro_ACC'] = metrics['ACC'].mean()
    return metrics


# def splitTrainingData(num_timesteps, split_props=[0.8, 0.1, 0.1]):
#     """
#     Split the dataset into training/validation/testing

#     Parameters
#     ----------
#     split_props : list
#         Each float value in the list indicates the proportion of the data to serve
#         as training, validation, and testing data. The values must sum to 1.
#     """
#     indices = np.arange(num_timesteps)
#     split_points = [int(num_timesteps*i) for i in split_props]
#     train_ix = np.random.choice(indices,
#                                 split_points[0],
#                                 replace=False)
#     val_ix = np.random.choice(list(set(indices)-set(train_ix)),
#                               split_points[1],
#                               replace=False)
#     test_ix = np.random.choice((list(set(indices)-set(train_ix)-set(val_ix))),
#                                split_points[2],
#                                replace=False)
#     return train_ix, val_ix, test_ix

