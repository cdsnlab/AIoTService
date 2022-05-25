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
        
    def call(self, X, is_train=True):
        if is_train: # Model chooses for itself during testing
            self.Controller._epsilon = self._epsilon # set explore/exploit trade-off
        else:
            self.Controller._epsilon = 0.0
        B, T, V = X.shape # Input shape (BATCH x TIMESTEPS x VARIABLES)     
        
        hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
        # halt_points = -np.ones(shape=(B, 1))
        halt_points = -tf.ones([B,1])
        predictions = tf.zeros([B, self.args.nclasses])
        
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        baselines = [] # Predicted baselines
        raw_probs = []
        pred_y = []
        
        for t in range(T):
            x = X[:,t,:]
            output, hidden = self.LSTM(x, states=hidden)
            
            # predict logits for all elements in the batch
            logits = self.out(output)
            yhat_t = tf.argmax(logits, 1)
            yhat_t = tf.reshape(yhat_t, (-1, 1))
            
            # compute halting probability, sample an action, and baseline
            time = tf.ones([B,1]) * t
            c_in = tf.stop_gradient(tf.concat([output, time], axis=1))
            # c_in = tf.concat([output, time], axis=1)
            a_t, p_t, w_t, probs_t = self.Controller(c_in)
            b_t = self.BaselineNetwork(c_in)
            
            # If a_t == 1 and this class hasn't been halted, save its logits
            predictions = tf.where((a_t == 1) & (predictions == 0), logits, predictions)
            probs_t = tf.where(halt_points == -1, probs_t, -1)
            yhat_t = tf.where(halt_points == -1, yhat_t, -1)
            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = tf.where((halt_points == -1) & (a_t == 1), time, halt_points)
            
            actions.append(a_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            baselines.append(b_t)
            raw_probs.append(probs_t)
            pred_y.append(yhat_t)
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
        
        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = np.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[b, :(1 + int(halt_points[b, 0]))] = 1
        return logits




# actions = []
# actions.append(action)
# actions = tf.concat(actions, axis=1)
# grad_mask = np.zeros_like(actions)

# grad_mask[0, :(1 + halt_points[0, 0])] = 1
# grad_mask[0, :1.]
# halt_points = np.array([[3], [5], [2]])


# t = 3
# B = 4
# Controller = Controller(args)
# logits = tf.random.normal([B, 2])
# predictions = tf.zeros([B, 2])
# halt_points = -np.ones(shape=(B, 1))
# time = tf.ones([B,1]) * t

# c_in = tf.stop_gradient(tf.concat([logits, time], axis=1))
# a_t, p_t, w_t = Controller(c_in)

# halt_points.numpy() +1



# log_pi = []
# log_pi.append(m.log_prob(action))

# tf.concat(log_pi, axis=1)

# probs = np.array([[1], [0.5], [0.0]])
# m = tfp.distributions.Bernoulli(probs=probs)
# action = m.sample(seed=0)
# action
# m.log_prob(action)
# m.log_prob([[0],[0],[0]])


# -tf.math.log(0.0)

