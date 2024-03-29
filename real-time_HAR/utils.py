import os
import argparse

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

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp_info_file", type=str, default="exp_info", help="Where to save the model once it is trained.")
    # Dataset hyperparameters
    parser.add_argument("--dataset", type=str, default="milan", help="Which dataset will be used")
    parser.add_argument("--seq_len", type=int, default=2000, help="The number of timesteps")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle training dataset or not.")
    parser.add_argument("--offset", type=int, default=20, help="The offset of the detected segmented windows")
    parser.add_argument("--expiration_period", type=int, default=-1, help="The expiration period of the sensor activation")
    parser.add_argument("--window_size", type=int, default=1, help="The window size for Lapras dataset")
    parser.add_argument("--noise_ratio", type=int, default=0, help="The ratio of the noise in the detected segmented windows")
    parser.add_argument("--random_noise", type=str2bool, default=False, help="Add noise to the beginning randomly.")
    parser.add_argument("--nsplits", type=int, default=3, help="The number of splits for validation")
    parser.add_argument("--nseries", type=int, default=0, help="The number of time series")
    parser.add_argument("--segmented", type=str2bool, default=False, help="Whether the activity episodes are segmented correctly")
    parser.add_argument("--with_other", type=str2bool, default=False, help="Whether Other class is going to be included in the dataset or not")
    parser.add_argument("--balance", type=str2bool, default=False, help="Whether some classes are balanced")
    parser.add_argument("--except_all_other_events", type=str2bool, default=False, help="Exclude all events for Other class")
    parser.add_argument("--noise_test_index", type=int, default=-1, help="The length of prefix length of dataset")
    parser.add_argument("--noise_low", type=int, default=-1, help="The lower bound of the amount of the noise")
    parser.add_argument("--noise_high", type=int, default=-1, help="The upper bound of the amount of the noise")
    parser.add_argument("--remove_short", type=str2bool, default=False, help="remove short episodes")
    parser.add_argument("--aug_multiple", type=int, default=0, help="The number of augmented examples per real one")
    parser.add_argument("--window_ratio", type=float, default="0.1", help="Ratio of window warping.")
    # Model hyperparameters
    parser.add_argument("--nhid", type=int, default=64, help="Number of dimensions of the hidden state of EARLIEST")
    parser.add_argument("--lam", type=float, default=0.08, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")
    parser.add_argument("--dropout_rate", type=float, default="0.5", help="Dropout rate.")
    parser.add_argument("--reg_rate", type=float, default="0.001", help="regularizer rate.")
    parser.add_argument("--_epsilon", type=float, default="0.1", help="epsilon for exploration/exploitation.")
    parser.add_argument("--model", type=str, default="EARLIEST", help="Which model will be used")
    parser.add_argument("--filter_name", type=str, default="attn", help="Which model will be used")
    parser.add_argument("--test_t", type=str2bool, default=False, help="test")
    parser.add_argument("--dff", type=int, default=64, help="The number of neurons of feed-forward network in the transformer encoder")
    parser.add_argument("--num_encoder", type=int, default=1, help="The number of encoders in the transformer structure")
    parser.add_argument("--num_heads", type=int, default=1, help="The number of heads in the transformer structure")
    parser.add_argument("--entropy_halting", type=str2bool, default=False, help="entropy and class membership prob are considered when calculating halting probability")
    parser.add_argument("--hidden_as_input", type=str2bool, default=False, help="entropy and class membership prob are considered when calculating halting probability")
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--nepochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.")
    parser.add_argument("--train_filter", type=str2bool, default=False, help="Train filter module additionally to avoid cold start problem")
    parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")
    parser.add_argument("--device", type=str, default="0", help="Which device will be used")
    parser.add_argument("--exp_num", type=str, default="0", help="Experiment number")
    parser.add_argument("--exp_id", type=str, default="0", help="Experiment identifier")
    parser.add_argument("--gamma", type=int, default=0, help="gamma for focal loss.")
    parser.add_argument("--class_weight", type=str2bool, default=False, help="Apply class weight or not")
    parser.add_argument("--decay_weight", type=float, default=10, help="decay weight for exploration")
    parser.add_argument("--n_fold_cv", type=str2bool, default=True, help="whether to conduct n-fold cross validation")
    parser.add_argument("--train", type=str2bool, default=True, help="whether to train the model")
    parser.add_argument("--test", type=str2bool, default=False, help="whether to test the model")
    parser.add_argument("--model_dir", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
    parser.add_argument("--full_seq", type=str2bool, default=False, help="Utilze the full sequence of the stream or not")
    return parser.parse_args()


