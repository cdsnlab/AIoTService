import os
import time
import random
import pickle
import logging
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


args = utils.create_parser()

def loss_EARLIEST(model, x, true_y, length, tr_points):  # shape of true_y is (B,)
    pred_logit = model(x, true_y, is_train=True, length=length, tr_points=tr_points)  # shape of pred_y is (B * nclasses)
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
    if args.model == "DETECTOR":
        model.halt_probs = model.halt_probs * model.grad_mask
    wait_penalty = tf.reduce_mean(tf.reduce_sum(model.halt_probs, 1))
    loss = loss_r + loss_b + loss_c + args.lam*(wait_penalty)
    if args.train_filter:
        CE_filter = tf.keras.losses.SparseCategoricalCrossentropy()
        tr_points = np.reshape(tr_points, (model.filter_logits.shape[0], -1))
        # true_tr = np.zeros((model.filter_logits.shape[0], args.offset))
        # for b in range(model.filter_logits.shape[0]):
        #     true_tr[b, tr_points[b][0]:] = 1
        loss_filter = CE_filter(y_true=tr_points, y_pred=model.filter_logits) # Classification loss
        loss += loss_filter*(model._epsilon)
        # loss += loss_filter*(model._epsilon)
    # if args.model == "PROPOSED":
    #     loss += model.loss_r_filter
    return loss, pred_logit, model.locations

def grad(model, x, true_y, length, tr_points):
    with tf.GradientTape() as tape:
        total_loss, pred_logit, model.locations = loss_EARLIEST(model, x, true_y, length, tr_points)
        # if args.model == "EARLIEST":
        # elif args.model == "basic":
    return total_loss, tape.gradient(total_loss, model.trainable_variables), pred_logit, model.locations

def train_step(model, x, true_y, length, tr_points):
    total_loss, grads, pred_logit, locations = grad(model, x, true_y=true_y, length=length, tr_points=tr_points)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(total_loss)
    train_accuracy(np.reshape(true_y, [-1,1]), pred_logit)

def test_step(model, x, true_y, length, num_event, tr_points):
    # if args.model == "EARLIEST":
    prev_ts = time.time()
    pred_logit = model(x, true_y, is_train=False, length=length, tr_points=tr_points)
    if args.test:
        duration = time.time() - prev_ts
        list_duration.append(duration)
    test_accuracy(np.reshape(true_y, [-1,1]), pred_logit)
    test_earliness(model.locations.flatten()/length)
    true_labels.append(true_y)
    pred_labels.append(tf.argmax(pred_logit, 1))
    list_locations.append(model.locations.flatten())
    list_filter_flags.append(model.filter_flags.flatten())
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
    list_distribution.append(model.distribution)
    list_attn.append(model.attention_weights)
    if args.model == "DETECTOR":
        list_estimated_tr.append(model.estimated_tr.flatten())
    # if args.model == 'CNN':
    #     list_cnn_feature.append(model.feature_map)
    #     list_attn.append(model.attn_encoder.attention_weights)
    # else:
    #     list_attn.append([])

def write_test_summary(true_y, pred_y):
    precision, recall, f1, support = f_score(true_y, pred_y, average=None, labels=range(args.nclasses))
    mac_precision, mac_recall, mac_f1, _ = f_score(true_y, pred_y, average='macro')
    mic_precision, mic_recall, mic_f1, _ = f_score(true_y, pred_y, average='micro')
    locations = np.concatenate(list_locations)
    filter_flags = np.concatenate(list_filter_flags)
    lengths = np.concatenate(list_lengths)
    event_count = np.concatenate(list_event_count)
    raw_probs = np.concatenate(list_probs)
    all_yhat = np.concatenate(list_yhat)
    all_dist = np.concatenate(list_distribution)
    all_attn = np.concatenate(list_attn)
    
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
        tf.summary.scalar('whole_earliness2', locations.mean() / lengths.mean(), step=epoch)
        hm = (2 * (1 - test_earliness.result()) * test_accuracy.result()) / ((1 - test_earliness.result()) + test_accuracy.result())
        tf.summary.scalar('whole_harmonic_mean', hm, step=epoch)
        
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
            tf.summary.scalar('by_cls_earliness2_mean', locations[idx].mean() / lengths[idx].mean(), step=epoch)
    
    # df = pd.DataFrame({'true_y': true_y, 'pred_y': pred_y, 'locations': locations, 'lengths': lengths, 'event_count': event_count})
    # df.to_csv(logdir + "/test_results.csv", index=False, encoding='utf=8')
    dict_analysis = {"idx":test_loader.indices, "raw_probs": raw_probs, "all_yhat": all_yhat, "true_y": true_y, 
                     "all_dist": all_dist, "noise_amount": data.noise_amount[test_loader.indices], "attn_scores": all_attn,
                     'pred_y': pred_y, 'locations': locations, 'lengths': lengths, 'event_count': event_count, "filter_flags": filter_flags}
    if args.model == "DETECTOR":
        dict_analysis['threshold_mse_list'] = model.mse_list
        dict_analysis['threshold'] = model.detector_threshold
        dict_analysis['estimated_tr'] = np.concatenate(list_estimated_tr)
        
    if args.test:
        dict_analysis['duration'] = list_duration
    with open(logdir + '/dict_analysis.pickle', 'wb') as f:
        pickle.dump(dict_analysis, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    if args.model == "DETECTOR":
        logger = tf.get_logger()
        logger.setLevel(logging.ERROR)
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
    args.N_FEATURES = data.N_FEATURES
    args.noise_amount = data.noise_amount
    kfold = StratifiedKFold(n_splits=args.nsplits, random_state=args.random_seed, shuffle=args.shuffle)
    print(args)
    
    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    train_loss = tf.keras.metrics.Mean('train_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    test_earliness = tf.keras.metrics.Mean('test_earliness', dtype=tf.float32)
    if args.train:
        for k, (train_idx, test_idx) in enumerate(kfold.split(data.X, data.Y)):
            # Define data loaders and model
            train_loader = Dataloader(train_idx, data.X, data.Y, data.lengths, data.event_counts, args.batch_size, shuffle=args.shuffle, tr_points=data.noise_amount)
            test_loader = Dataloader(test_idx, data.X, data.Y, data.lengths, data.event_counts, args.batch_size, shuffle=args.shuffle, tr_points=data.noise_amount)
            # if args.model in ["EARLIEST", "ATTENTION"]:
            if args.model == 'DETECTOR':
                temp_model = args.model
                args.model = "ATTENTION"
                args.test_idx = test_idx
                detector = EARLIEST(args)
                detector._epsilon = 0
                list_avg_duration = []
                temp = np.array([[3]])
                detector(np.reshape(data.X[0], (1, -1, data.N_FEATURES)), temp, length=temp, is_train=False)
                detector.load_weights(os.path.join(args.model_dir, f'fold_{k+1}', 'model'))
                detector.attn_encoder(data.X[args.test_idx, :args.offset, :], is_train=False)
                args.detector = detector.attn_encoder
                args.model = temp_model
                del detector
            model = EARLIEST(args)
                
            # Define summary writer
            logdir = "./output/log/" + curr_time + f'/fold_{k+1}'
            print(f'tensor board dir: {logdir}')
            train_summary_writer = tf.summary.create_file_writer(logdir + '/train')
            test_summary_writer = tf.summary.create_file_writer(logdir + '/test')
            cls_summary_writer = {i:tf.summary.create_file_writer(logdir + f'/cls_{data.idx2label[i]}') for i in range(args.nclasses)}
            for epoch in tqdm(range(args.nepochs)):
                model._epsilon = exponentials[epoch]
                for x, true_y, length, _, tr_points in train_loader:
                    train_step(model, x, true_y, length, tr_points)
                # end of epoch
                with train_summary_writer.as_default():
                    tf.summary.scalar('whole_accuracy', train_accuracy.result(), step=epoch)
                print("k{}_epoch {:03d}:  train_loss: {:.3}  train_acc: {:.3}".format(k+1, epoch, train_loss.result(), train_accuracy.result()))
                
                # Run test for every 10 epochs
                if (epoch + 1) % 10 == 0:
                    true_labels, pred_labels, list_locations, list_lengths, list_event_count = [], [], [], [], []
                    list_probs, list_yhat, list_distribution, list_attn, list_filter_flags = [], [], [], [], []
                    list_estimated_tr = []
                    # list_cnn_feature = []
                    for x, true_y, length, num_event, tr_points in test_loader: 
                        test_step(model, x, true_y, length, num_event, tr_points)
                    # Write summary
                    write_test_summary(np.concatenate(true_labels), np.concatenate(pred_labels))
                    print("Validation Accuracy: {:.3}".format(test_accuracy.result()))
                    print("Mean proportion used: {:.3}".format(test_earliness.result()))
                    print("Harmonic Mean: {:.3}".format((2 * (1 - test_earliness.result()) * test_accuracy.result()) / ((1 - test_earliness.result()) + test_accuracy.result())))
                    print("The percentage of the filter used: {:.3}".format(np.concatenate(list_filter_flags).mean()))
                # Reset states of metrics
                train_accuracy.reset_states()
                train_loss.reset_states()
                test_accuracy.reset_states()
                test_earliness.reset_states()
            model.save_weights(os.path.join(logdir, 'model'))
            if not args.n_fold_cv:
                break
        print(f'tensor board dir: {logdir}')
        f = open(f"./exp_info/{args.exp_info_file}.txt", 'a')
        f.write(f"{args.dataset}\t{args.lam}\t{args.model}\t{logdir}\n")
        f.close()
            
    if args.test:
        epoch = 0
        model = EARLIEST(args)
        model._epsilon = 0
        list_avg_duration = []
        temp = np.array([[3]])
        model(np.reshape(data.X[0], (1, -1, data.N_FEATURES)), temp, length=temp, is_train=False)
        for k in range(args.nsplits):
            model.load_weights(os.path.join(args.model_dir, f'fold_{k+1}', 'model'))
            with open(os.path.join(args.model_dir, f'fold_{k+1}/dict_analysis.pickle'), 'rb') as f:
                dict_analysis = pickle.load(f)
            test_loader = Dataloader(dict_analysis["idx"], data.X, data.Y, data.lengths, data.event_counts, args.batch_size, shuffle=args.shuffle, tr_points=data.noise_amount)
            true_labels, pred_labels, list_locations, list_lengths, list_event_count = [], [], [], [], []
            list_probs, list_yhat, list_distribution, list_attn, list_filter_flags = [], [], [], [], []
            list_duration = []
            
            logdir = "./output/log/" + curr_time + f'/fold_{k+1}'
            print(f'tensor board dir: {logdir}')
            test_summary_writer = tf.summary.create_file_writer(logdir + '/test')
            cls_summary_writer = {i:tf.summary.create_file_writer(logdir + f'/cls_{data.idx2label[i]}') for i in range(args.nclasses)}
            for x, true_y, length, num_event, tr_points in test_loader: 
                test_step(model, x, true_y, length, num_event, tr_points)
            # for x, true_y, length, num_event in test_loader: 
            #     test_step(model, x, true_y, length, num_event)
            # Write summary
            write_test_summary(np.concatenate(true_labels), np.concatenate(pred_labels))
            print("Validation Accuracy: {:.3}".format(test_accuracy.result()))
            print("Mean proportion used: {:.3}".format(test_earliness.result()))
            print("Harmonic Mean: {:.3}".format((2 * (1 - test_earliness.result()) * test_accuracy.result()) / ((1 - test_earliness.result()) + test_accuracy.result())))
            avg_duration = np.mean(np.array(list_duration) / np.array(list_locations))
            list_avg_duration.append(avg_duration)
            print(f"Duration: {avg_duration}")
            test_accuracy.reset_states()
            test_earliness.reset_states()
            if not args.n_fold_cv:
                break
        f = open(f"./exp_info/{args.exp_info_file}.txt", 'a')
        f.write(f"{args.dataset}\t{args.model}\t{np.mean(list_avg_duration)}\t{args.model_dir}\t./output/log/{curr_time}\n")
        f.close()
