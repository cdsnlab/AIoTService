import os

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    np.nan_to_num(metrics["TPR"], copy=False)
    np.nan_to_num(metrics["TNR"], copy=False)
    np.nan_to_num(metrics["PPV"], copy=False)
    np.nan_to_num(metrics["NPV"], copy=False)
    np.nan_to_num(metrics["FPR"], copy=False)
    np.nan_to_num(metrics["FNR"], copy=False)
    np.nan_to_num(metrics["FDR"], copy=False)
    np.nan_to_num(metrics["ACC"], copy=False)
    metrics['F1'] = (metrics["PPV"] * metrics["TPR"]) / (metrics["PPV"] + metrics["TPR"]) * 2 # F1-score
    np.nan_to_num(metrics["F1"], copy=False)
    
    metrics['micro_TPR'] = (metrics['TPR'] * support).sum() / support.sum()
    metrics['micro_TNR'] = (metrics['TNR'] * support).sum() / support.sum()
    metrics['micro_PPV'] = (metrics['PPV'] * support).sum() / support.sum()
    metrics['micro_NPV'] = (metrics['NPV'] * support).sum() / support.sum()
    metrics['micro_FPR'] = (metrics['FPR'] * support).sum() / support.sum()
    metrics['micro_FNR'] = (metrics['FNR'] * support).sum() / support.sum()
    metrics['micro_FDR'] = (metrics['FDR'] * support).sum() / support.sum()
    metrics['micro_ACC'] = (metrics['ACC'] * support).sum() / support.sum()
    metrics['micro_F1'] = (metrics['F1'] * support).sum() / support.sum()
    
    metrics['macro_TPR'] = metrics['TPR'].mean()
    metrics['macro_TNR'] = metrics['TNR'].mean()
    metrics['macro_PPV'] = metrics['PPV'].mean()
    metrics['macro_NPV'] = metrics['NPV'].mean()
    metrics['macro_FPR'] = metrics['FPR'].mean()
    metrics['macro_FNR'] = metrics['FNR'].mean()
    metrics['macro_FDR'] = metrics['FDR'].mean()
    metrics['macro_ACC'] = metrics['ACC'].mean()
    metrics['macro_F1'] = metrics['F1'].mean()
    return metrics

def plot_confusion_matrix(true_y, pred_y, dir, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    # accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy
    cm = confusion_matrix(true_y, pred_y)
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, ha="right")
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.show()
    plt.savefig(dir)
    plt.clf()


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

