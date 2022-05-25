import os
import random
import pickle
import argparse
from math import sqrt
from datetime import datetime

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



def loss_EARLIEST(model, x, true_y):  # shape of true_y is (B,)
    pred_logit = model(x, is_train=True)  # shape of pred_y is (B * nclasses)
    pred_y = tf.argmax(pred_logit, 1)
    if args.class_weight:
        class_weight = [1 / sqrt(list(true_y).count(i)) if list(true_y).count(i) != 0 else 0 for i in range(args.nclasses)]
    else:
        class_weight = None
    
    # --- compute reward ---
    r = tf.stop_gradient(tf.where((pred_y == true_y), 1.0, -1.0))
    r = tf.reshape(r, [-1, 1])
    R = r * model.grad_mask
    
    # --- rescale reward with baseline ---
    b = model.baselines * model.grad_mask
    adjusted_reward = R - tf.stop_gradient(b)
    
    # --- compute losses ---
    MSE = tf.keras.losses.MeanSquaredError()
    if args.gamma == 0:
        CE = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        CE = SparseCategoricalFocalLoss(gamma=args.gamma, class_weight=class_weight)
    loss_b = MSE(b, R) # Baseline should approximate mean reward  # 엄밀히 따지면 인스턴스간 길이가 다르기 때문에 인스턴스별 평균을 구하고 배치의 평균을 구해야되는 것 같음 
    loss_r = tf.reduce_sum(-model.log_pi*adjusted_reward) / model.log_pi.shape[1] # RL loss  # 이것도 위와 마찬가지. 근데 어찌 되었든 배치의 평균을 구하기는 했음.
    loss_c = CE(y_true=true_y, y_pred=pred_logit) # Classification loss
    wait_penalty = tf.reduce_mean(tf.reduce_sum(model.halt_probs, 1))
    loss = loss_r + loss_b + loss_c + args.lam*(wait_penalty)
    return loss, pred_logit, model.locations

def grad(model, x, true_y):
    with tf.GradientTape() as tape:
        total_loss, pred_logit, model.locations = loss_EARLIEST(model, x, true_y)
        # if args.model == "EARLIEST":
        # elif args.model == "basic":
    return total_loss, tape.gradient(total_loss, model.trainable_variables), pred_logit, model.locations

def train_step(model, x, true_y):
    total_loss, grads, pred_logit, locations = grad(model, x, true_y=true_y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(total_loss)
    train_accuracy(np.reshape(true_y, [-1,1]), pred_logit)        

def test_step(model, x, true_y, length, num_event):
    if args.model == "EARLIEST":
        pred_logit = model(x, is_train=False)
        test_accuracy(np.reshape(true_y, [-1,1]), pred_logit)
        test_earliness(model.locations.flatten()/length)
        true_labels.append(true_y)
        pred_labels.append(tf.argmax(pred_logit, 1))
        list_locations.append(model.locations.flatten())
        list_lengths.append(length)
        list_event_count.append(np.take_along_axis(num_event, model.locations.astype(np.int16)-1, axis=1).flatten())
        # Example:
            # num_event = np.array([[1,1,2,3,4],
            #                       [1,4,7,7,10],
            #                       [1,2,9,35,45]])    
            # locations = np.array([[1],
            #                       [3],
            #                       [4]])
        list_probs.append(model.raw_probs)
        list_yhat.append(model.pred_y)

def write_test_summary(true_y, pred_y):
    precision, recall, f1, support = f_score(true_y, pred_y, average=None, labels=range(args.nclasses))
    mac_precision, mac_recall, mac_f1, _ = f_score(true_y, pred_y, average='macro')
    mic_precision, mic_recall, mic_f1, _ = f_score(true_y, pred_y, average='micro')
    locations = np.concatenate(list_locations)
    lengths = np.concatenate(list_lengths)
    event_count = np.concatenate(list_event_count)
    raw_probs = np.concatenate(list_probs)
    all_yhat = np.concatenate(list_yhat)
    
    # Calculate representative metric of whole data
    with test_summary_writer.as_default():
        tf.summary.scalar('whole_accuracy', test_accuracy.result(), step=epoch)
        tf.summary.scalar('whole_earliness', test_earliness.result(), step=epoch)
        tf.summary.scalar('whole_macro_precision', mac_precision, step=epoch)
        tf.summary.scalar('whole_macro_recall', mac_recall, step=epoch)
        tf.summary.scalar('whole_macro_f1', mac_f1, step=epoch)
        tf.summary.scalar('whole_micro_precision', mic_precision, step=epoch)
        tf.summary.scalar('whole_micro_recall', mic_recall, step=epoch)
        tf.summary.scalar('whole_micro_f1', mic_f1, step=epoch)
        tf.summary.scalar('whole_location_mean', locations.mean(), step=epoch)
        tf.summary.scalar('whole_count_mean', event_count.mean(), step=epoch)
        
    # Calculate metrics by classes
    for i, summary_writer in cls_summary_writer.items():
        idx = np.where(true_y == i)
        with summary_writer.as_default():
            tf.summary.scalar('by_cls_precision', precision[i], step=epoch)
            tf.summary.scalar('by_cls_recall', recall[i], step=epoch)
            tf.summary.scalar('by_cls_f1', f1[i], step=epoch)
            tf.summary.scalar('by_cls_support', support[i], step=epoch)
            tf.summary.scalar('by_cls_location_mean', locations[idx].mean(), step=epoch)
            tf.summary.scalar('by_cls_location_std',  locations[idx].std(), step=epoch)
            tf.summary.scalar('by_cls_earliness_mean',(locations[idx] / lengths[idx]).mean(), step=epoch)
            tf.summary.scalar('by_cls_earliness_std', (locations[idx] / lengths[idx]).std(), step=epoch)
            tf.summary.scalar('by_cls_event_count_mean', event_count[idx].mean(), step=epoch)
            tf.summary.scalar('by_cls_event_count_std',  event_count[idx].std(), step=epoch)
    
    df = pd.DataFrame({'true_y': true_y, 'pred_y': pred_y, 'locations': locations, 'lengths': lengths, 'event_count': event_count})
    df.to_csv(logdir + "/test_results.csv", index=False, encoding='utf=8')
    dict_analysis = {"idx":test_loader.indices, "raw_probs": raw_probs, "all_yhat": all_yhat, "true_y": true_y}
    with open(logdir + '/dict_analysis.pickle', 'wb') as f:
        pickle.dump(dict_analysis, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
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

    # Load dataset
    if args.segmented == True:
        data = CASAS_RAW_SEGMENTED(args)
    elif args.segmented == False:
        data = CASAS_RAW_NATURAL(args)
    # elif args.dataset == "adlmr":
    #     data = CASAS_ADLMR(args)
    args.nclasses = data.N_CLASSES
    kfold = StratifiedKFold(n_splits=args.nsplits, random_state=args.random_seed, shuffle=args.shuffle)
    
    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    train_loss = tf.keras.metrics.Mean('train_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    test_earliness = tf.keras.metrics.Mean('test_earliness', dtype=tf.float32)
    for k, (train_idx, test_idx) in enumerate(kfold.split(data.X, data.Y)):
        # Define data loaders and model
        train_loader = Dataloader(train_idx, data.X, data.Y, data.lengths, data.event_counts, args.batch_size, shuffle=args.shuffle)
        test_loader = Dataloader(test_idx, data.X, data.Y, data.lengths, data.event_counts, args.batch_size, shuffle=args.shuffle)
        if args.model == "EARLIEST":
            model = EARLIEST(args)
            
        # Define summary writer
        logdir = "./output/log/" + curr_time + f'/fold_{k+1}'
        print(f'tensor board dir: {logdir}')
        train_summary_writer = tf.summary.create_file_writer(logdir + '/train')
        test_summary_writer = tf.summary.create_file_writer(logdir + '/test')
        cls_summary_writer = {i:tf.summary.create_file_writer(logdir + f'/cls_{data.idx2label[i]}') for i in range(args.nclasses)}
        for epoch in tqdm(range(args.nepochs)):
            model._epsilon = exponentials[epoch]
            for x, true_y, length, _ in train_loader:
                train_step(model, x, true_y)
            # end of epoch
            with train_summary_writer.as_default():
                tf.summary.scalar('whole_accuracy', train_accuracy.result(), step=epoch)
            print("k{}_epoch {:03d}:  train_loss: {:.3}  train_acc: {:.3}".format(k+1, epoch, train_loss.result(), train_accuracy.result()))
            
            # Run test for every 10 epochs
            if (epoch + 1) % 10 == 0:
                true_labels, pred_labels, list_locations, list_lengths, list_event_count = [], [], [], [], []
                list_probs, list_yhat = [], []
                for x, true_y, length, num_event in test_loader: 
                    test_step(model, x, true_y, length, num_event)
                # Write summary
                write_test_summary(np.concatenate(true_labels), np.concatenate(pred_labels))
                print("Validation Accuracy: {:.3}".format(test_accuracy.result()))
                print("Mean proportion used: {:.3}".format(test_earliness.result()))
            # Reset states of metrics
            train_accuracy.reset_states()
            train_loss.reset_states()
            test_accuracy.reset_states()
            test_earliness.reset_states()
        tf.keras.models.save_model(model, logdir, save_format='tf', save_traces=False)
        print(f'tensor board dir: {logdir}')
        # break

