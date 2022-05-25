# Load data frame from tensor board.dev --------------------------------------------------
import tensorboard as tb

experiment_id = "lL2XetbbT3ao7kfYj0ie2A"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df

dfw = experiment.get_scalars(pivot=True) 
dfw


# Metric --------------------------------------------------
import numpy as np
import tensorflow as tf
m = tf.keras.metrics.Mean()
np.round(m.result(), 1)
m.reset_states()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

test_accuracy(np.reshape(np.array([1,2,3]), [-1,1]), np.reshape(np.array([1,2,3]), [-1,1]))
test_accuracy.reset_states()
pred_logit = tf.random.normal([4, 3])
pred_logit

np.concatenate(val_true_y)



data.label2idx

test_precision = {}
{i:tf.keras.metrics. for i in range(args.nclasses)}


true_y = [0, 0, 1, 0]
pred_y = [0, 0, 1, 2]
pred_logit = tf.random.normal([4, 3])
pred_logit = tf.math.softmax(pred_logit, 1)

m = tf.keras.metrics.Precision(class_id=0, name = "precision_0")
m(np.reshape(true_y, [-1, 1]), np.reshape(pred_y, [-1,1]))
# m(true_y, pred_y)
m.result().numpy()



from sklearn.metrics import precision_recall_fscore_support
cls_summary_writer = {i:tf.summary.create_file_writer(logdir + f'/cls_{data.idx2label[i]}') for i in range(args.nclasses)}



precision, recall, f1, support = precision_recall_fscore_support(true_y, pred_y, average=None, labels=range(args.nclasses))
mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(true_y, pred_y, average='macro')
mic_precision, mic_recall, mic_f1, _ = precision_recall_fscore_support(true_y, pred_y, average='micro')

for i, summary_writer in cls_summary_writer.items():
    with summary_writer.as_default():
        tf.summary.scalar('precision', precision[i], step=epoch)
        tf.summary.scalar('recall', recall[i], step=epoch)
        tf.summary.scalar('f1', f1[i], step=epoch)
        tf.summary.scalar('support', support[i], step=epoch)


aa = []
aa.append(np.reshape(true_y, [-1, 1]).flatten())

np.concatenate(aa)



class_weight = [1 / list(true_y).count(i) if list(true_y).count(i) != 0 else 0 for i in range(args.nclasses)]

tf.convert_to_tensor(class_weight, dtype=tf.dtypes.float32)


def record_num_event(self, x):
    prev_state = np.zeros((self.N_FEATURES))
    count = 0
    event_count = []
    for i, row in enumerate(test_x):
        if i < test_length:
            count += (row != prev_state).sum()
            prev_state = row.copy()
        event_count.append(count)
        
        
y_true = [1, 1, 2]
y_pred = [[0.0, 1.0, 0], [0.0, 1.0, 0], [0.1, 0.8, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.
scce = tf.keras.losses.SparseCategoricalCrossentropy()
scce(y_true, y_pred).numpy()

true_y, pred_y
locations = np.concatenate(list_locations)
    lengths = np.concatenate(list_lengths)
    event_count = np.concatenate(list_event_count)


true_y = np.array([1,1,1,1,1])
pred_y = np.array([2,2,2,2,2])
locations = np.array([3,3,3,3,3])
lengths = np.array([4,4,4,4,4])
event_count = np.array([5,5,5,5,5])
    
import pandas as pd
df = pd.DataFrame({'true_y':true_y, 'pred_y': pred_y, 'locations': locations, 'lengths': lengths, 'event_count': event_count})
df.to_csv(logdir + "/test_results.csv", index=False, encoding='utf=8')


# logdir = "./output/log/" + "220506-193423" + f'/fold_{k+1}'