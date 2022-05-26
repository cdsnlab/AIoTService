import os
import random
import pickle
import argparse
from math import sqrt
from datetime import datetime
from tkinter import X

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support as f_score

import utils
from model import EARLIEST
from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset hyperparameters
parser.add_argument("--dataset", type=str, default="milan", help="Which dataset will be used")
parser.add_argument("--seq_len", type=int, default=2000, help="The number of timesteps")
parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle training dataset or not.")
# parser.add_argument("--nclasses", type=int, default=15, help="The number of classes.")
parser.add_argument("--nsplits", type=int, default=5, help="The number of splits for validation")
parser.add_argument("--nseries", type=int, default=0, help="The number of time series")
parser.add_argument("--prefix_len", type=int, default=10, help="The length of prefix length of dataset")
parser.add_argument("--remove_prefix", type=bool, default=False, help="Whether the prefix is removed")
parser.add_argument("--rnd_prefix", type=bool, default=False, help="Whether to add random events to the beginning of the activity")
parser.add_argument("--segmented", type=bool, default=True, help="Whether the activity episodes are segmented correctly")
# Model hyperparameters
parser.add_argument("--nhid", type=int, default=100, help="Number of dimensions of the hidden state of EARLIEST")
parser.add_argument("--lam", type=float, default=0.08, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")
parser.add_argument("--dropout_rate", type=float, default="0.2", help="Dropout rate.")
parser.add_argument("--reg_rate", type=float, default="0.001", help="regularizer rate.")
parser.add_argument("--_epsilon", type=float, default="0.1", help="epsilon for exploration/exploitation.")
parser.add_argument("--model", type=str, default="EARLIEST", help="Which model will be used")
# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
parser.add_argument("--nepochs", type=int, default=50, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.")
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")
parser.add_argument("--device", type=str, default="0", help="Which device will be used")
parser.add_argument("--exp_num", type=str, default="0", help="Experiment number")
parser.add_argument("--gamma", type=int, default=0, help="gamma for focal loss.")
parser.add_argument("--class_weight", type=bool, default=False, help="Apply class weight or not")
parser.add_argument("--decay_weight", type=float, default=10, help="decay weight for exploration")


args = parser.parse_args()


tf.autograph.set_verbosity(0)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= args.device
os.environ['PYTHONHASHSEED'] = str(args.random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)
curr_time = datetime.now().strftime("%y%m%d-%H%M%S")
exponentials = utils.exponentialDecay(args.nepochs, args.decay_weight)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)



args.dataset = "cairo"
# data = CASAS_RAW_NATURAL(args)
data = CASAS_RAW_SEGMENTED(args)

data.state_matrix.shape
data.labels.shape


del model
logdir = './output/log/220525-205141/'


test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
acc = []
for k in range(5):
    args.nclasses = 7
    model = EARLIEST(args)
    model._epsilon = 0
    model(np.reshape(data.X[0], (1, -1, 27)), is_train=False )
    model.load_weights(os.path.join(logdir, f'fold_{k+1}', 'model'))
    
    with open(os.path.join(logdir, f'fold_{k+1}/dict_analysis.pickle'), 'rb') as f:
        dict_analysis = pickle.load(f)
        
    test_loader = Dataloader(dict_analysis["idx"], data.X, data.Y, data.lengths, data.event_counts, args.batch_size, shuffle=args.shuffle)
    for x, true_y, length, num_event in test_loader: 
        pred_logit = model(x, is_train=False)
        test_accuracy(np.reshape(true_y, [-1,1]), pred_logit)
    acc.append(test_accuracy.result().numpy())

