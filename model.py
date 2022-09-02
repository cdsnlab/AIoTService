import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BaselineNetwork(tf.keras.layers.Layer):
    """
    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """
    def __init__(self, args):
        super(BaselineNetwork, self).__init__()
        self.args = args
        
    def build(self, input_shape):
        self.fc = layers.Dense(1, activation=None)

    def call(self, x, is_train=True):
        b = self.fc(tf.stop_gradient(x))  # 인풋 x가 이미 모델과 끊간 경우 tf.stop_gradient 지워도됨
        return b
        # b = self.fc(x)
        # return b
    
class Controller(tf.keras.layers.Layer):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, args):
        super(Controller, self).__init__()
        self.args = args
        self._epsilon = self.args._epsilon
        
    def build(self, input_shape):
        self.fc = layers.Dense(1, activation='sigmoid')  # Optimized w.r.t. reward

    def call(self, x, is_train=True):
        probs = self.fc(x)
        probs = (1-self._epsilon)*probs + self._epsilon*0.05 # exploration for WAIT
        # probs = np.array([[0.8], [0.1], [0.4]])
        m = tfp.distributions.Bernoulli(probs=probs)
        action = m.sample(seed=self.args.random_seed)
        log_pi = m.log_prob(action)
        return action, log_pi, -tf.math.log(probs), probs  # shape: BATCH x 1
    
class EARLIEST(tf.keras.Model):
    def __init__(self, args):
        super(EARLIEST, self).__init__(name='')
        self.args = args
        self.Controller = Controller(self.args)
        self.BaselineNetwork = BaselineNetwork(self.args)
        self.LSTM = layers.LSTMCell(self.args.nhid)
        self.initial_states = tf.zeros([self.args.batch_size, self.args.nhid])
        self.out = layers.Dense(self.args.nclasses, activation='softmax')  # Discriminator
    
    def entropy(self, p):
        id_p = np.where(p != 0)
        return -np.sum(p[id_p]*np.log(p[id_p]))
        
    def call(self, X, y_true, is_train=True, pred_at=-1, length=None, noise_amount=0):
        if is_train: # Model chooses for itself during testing
            self.Controller._epsilon = self._epsilon # set explore/exploit trade-off
        else:
            self.Controller._epsilon = 0.0
        B, T, V = X.shape # Input shape (BATCH x TIMESTEPS x VARIABLES)     
        
        hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
        # halt_points = -np.ones(shape=(B, 1))
        halt_points = -tf.ones([B,1])
        predictions = tf.zeros([B, self.args.nclasses])
        if length is not None:
            length = length.reshape((-1, 1))
        
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        baselines = [] # Predicted baselines
        raw_probs = []
        pred_y = []
        distribution = []
        
        if not is_train and pred_at != -1:
            halt_points = pred_at / 100 * length + noise_amount
            for t in range(T):
                x = X[:,t,:]
                output, hidden = self.LSTM(x, states=hidden)
                
                # predict logits for all elements in the batch
                logits = self.out(output)
                predictions = tf.where((t > halt_points) & (predictions == 0), logits, predictions)
                if t > np.max(halt_points):
                    break
            logits = tf.where(predictions == 0.0, logits, predictions)
            self.locations = halt_points.round().astype(int) - noise_amount
            return logits
        
        for t in range(T):
            x = X[:,t,:]
            output, hidden = self.LSTM(x, states=hidden)
            
            # predict logits for all elements in the batch
            logits = self.out(output)     
            ent = -np.sum(logits*np.log(logits), axis=1).reshape(B, -1)
            # ent = np.array([self.entropy(np.array(d)) for d in logits]).reshape(B, -1)
            yhat_t = tf.argmax(logits, 1)
            yhat_t = tf.reshape(yhat_t, (-1, 1))
            
            # compute halting probability, sample an action, and baseline
            if self.args.test_t:
                t = self.t
            time = tf.ones([B,1]) * t
            c_in = tf.stop_gradient(tf.concat([output, time], axis=1))
            # c_in = tf.concat([output, time], axis=1)
            a_t, p_t, w_t, probs_t = self.Controller(c_in)
            b_t = self.BaselineNetwork(c_in)
            # if self.args.delay_halt and not is_train:
            if self.args.delay_halt:
                cls_4 = np.array([0, 1, 2, 3])
                y_true = np.reshape(y_true, (B, -1))
                target_class = np.where(y_true == cls_4, 1, 0).sum(axis=1).reshape(B, -1)
                a_t = tf.where((ent > self.args.entropy_threshold) & (target_class == 1), 0, a_t)
            
            if t < self.args.offset and self.args.read_all_tw:
                a_t = a_t * 0
            
            # If a_t == 1 and this class hasn't been halted, save its logits
            predictions = tf.where((a_t == 1) & (predictions == 0), logits, predictions)
            predictions = tf.where((length-1 == t) & (predictions == 0), logits, predictions)
            probs_t = tf.where(halt_points == -1, probs_t, -1)
            yhat_t = tf.where(halt_points == -1, yhat_t, -1)
            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = tf.where((halt_points == -1) & (a_t == 1), time, halt_points)
            halt_points = tf.where((halt_points == -1) & (length-1 == t), time, halt_points)
            
            actions.append(a_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            baselines.append(b_t)
            raw_probs.append(probs_t)
            pred_y.append(yhat_t)
            distribution.append(logits)
            if np.sum((halt_points == -1)) == 0:  # If no negative values, every class has been halted
                break
            # If one element in the batch has not been halting, use its final prediction
        logits = tf.where(predictions == 0.0, logits, predictions)
        halt_points = tf.where(halt_points == -1, time, halt_points)
        
        self.locations = halt_points.numpy() + 1
        self.actions = tf.concat(actions, axis=1)
        self.log_pi = tf.concat(log_pi, axis=1)
        self.halt_probs = tf.concat(halt_probs, axis=1)
        self.baselines = tf.concat(baselines, axis=1)
        self.raw_probs = pad_sequences(tf.concat(raw_probs, axis=1), padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len, value=-1)
        self.pred_y = pad_sequences(tf.concat(pred_y, axis=1), padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len, value=-1)
        
        distribution = tf.concat(distribution, axis=1)
        distribution = tf.reshape(distribution, (B, -1, self.args.nclasses))
        self.distribution = pad_sequences(distribution, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len, value=-1)
        
        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = np.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[b, :(1 + int(halt_points[b, 0]))] = 1
        return logits


    # def get_config(self):
    #     return {"hidden_units": self.hidden_units}

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


class SEGMENTATION(tf.keras.Model):
    def __init__(self, args):
        super(SEGMENTATION, self).__init__(name='')
        self.args = args
        self.LSTM = layers.LSTMCell(self.args.nhid)
        self.initial_states = tf.zeros([self.args.batch_size, self.args.nhid])
        self.out = layers.Dense(1, activation='sigmoid')  # Discriminator
        
    def call(self, X, tr_boundary, tr_point, lengths, is_train=True):
        B, T, V = X.shape # Input shape (BATCH x TIMESTEPS x VARIABLES)     
        hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
        # tr_predictions = -tf.ones([B, 1])
        # pred_tr_points = -tf.ones([B, 1])
        end_points = -tf.ones([B,1])
        
        pred_tr_list = []
        for t in range(T):
            x = X[:,t,:]
            output, hidden = self.LSTM(x, states=hidden)
            pred_tr = self.out(output)
            pred_tr_list.append(pred_tr)
            
            end_points = tf.where(((tr_point + lengths) < t), 1, end_points)
            if np.sum((end_points == -1)) == 0:  # If no negative values, every class has been halted
                break           
        pred_tr = tf.concat(pred_tr_list, axis=1)
        
        if is_train:
            offset = 1
            idx_pos_samples = [[i for i in range(p-offset, p+offset+1, 1) if i >= 0] for p in tr_point]
            idx_neg_samples = []
            for i, idx_pos_sam in enumerate(idx_pos_samples):
                range_pool = range(lengths[i] + tr_point[i]) if len(range(lengths[i] + tr_point[i])) < self.args.seq_len * 2 else range(self.args.seq_len * 2)
                pool = set(range_pool) - set(idx_pos_sam)
                replace = False if (len(pool) >= offset*2) else True
                idx_neg_samples.append(list(np.random.choice(list(pool), offset*2, replace=replace)))
            idx_pred = [a+b for a,b in zip(idx_pos_samples, idx_neg_samples)]
            idx_pred = tf.convert_to_tensor(idx_pred, dtype=tf.int32)
            pred_tr = tf.experimental.numpy.take_along_axis(pred_tr, idx_pred, axis=1)
            true_tr = tf.experimental.numpy.take_along_axis(tr_boundary, idx_pred, axis=1)
            return pred_tr, true_tr
        else:
            return pred_tr, tr_boundary
        
        
        
# args.dataset = "milan"
# data = CASAS_RAW_NATURAL(args)
# model = SEGMENTATION(args)

# x = data.X[:10]
# tr_b = data.gt_boundary[:10]
# tr_point = data.tr_points[:10]
# lengths = data.lengths[:10]

# pred_tr, true_tr = model(x, tr_b, tr_point, lengths, is_train=True)

# offset = 1
# idx_pos_samples = [[i for i in range(p-offset, p+offset+1, 1) if i >= 0] for p in tr_point]

# idx_neg_samples = []
# for i, idx_pos_sam in enumerate(idx_pos_samples):
#     range_pool = range(lengths[i] + tr_point[i]) if len(range(lengths[i] + tr_point[i])) < model.args.seq_len * 2 else range(model.args.seq_len * 2)
#     pool = set(range_pool) - set(idx_pos_sam)
#     replace = False if (len(pool) >= offset*2) else True
#     idx_neg_samples.append(list(np.random.choice(list(pool), offset*2, replace=replace)))
# idx_pred = [a+b for a,b in zip(idx_pos_samples, idx_neg_samples)]
# idx_pred = tf.convert_to_tensor(idx_pred, dtype=tf.int32)
# pred_tr = tf.experimental.numpy.take_along_axis(model.pred_tr, idx_pred, axis=1)
# true_tr = tf.experimental.numpy.take_along_axis(tr_b, idx_pred, axis=1)




# np.take_along_axis(np.array(model.pred_tr), idx_pred, axis=1)


        
            
            
            #  # If a_t == 1 and this class hasn't been halted, save its logits
            # pred_tr_points = tf.where((pred_tr_points == -1) & (logits >= 0.5), t, pred_tr_points)
            # tr_predictions = tf.where((pred_tr_points != -1) & (a_t == 1), logits, tr_predictions)
            
            # actions.append(a_t)
            # log_pi.append(p_t)

