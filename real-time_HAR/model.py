import sys
import random
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

class Controller_ADAPTIVE(tf.keras.layers.Layer):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, args):
        super(Controller_ADAPTIVE, self).__init__()
        self.args = args
        self._epsilon = self.args._epsilon
        
    def build(self, input_shape):
        self.fc = layers.Dense(3, activation='softmax')  # Optimized w.r.t. reward

    def call(self, x, is_train=True):
        # probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        if random.random() > self._epsilon:
            probs = self.fc(x)
        else:
            probs = tf.ones([x.shape[0], 3]) / 3
        m = tfp.distributions.Categorical(probs=probs)
        action = m.sample(seed=self.args.random_seed)
        log_pi = m.log_prob(action)
        action = tf.reshape(action, (-1, 1))
        log_pi = tf.reshape(log_pi, (-1, 1))
        halt_probs = tf.reshape(probs[:, 1], (-1, 1))
        return action, log_pi, -tf.math.log(halt_probs), halt_probs  # shape: BATCH x 1    
    
class EARLIEST(tf.keras.Model):
    def __init__(self, args):
        super(EARLIEST, self).__init__(name='')
        self.args = args
        if self.args.model == 'ATTENTION':
            self.attn_encoder = TransformerEncoder(self.args)
        if self.args.model == 'ADAPTIVE':
            self.Controller = Controller_ADAPTIVE(self.args)
            if self.args.filter_name == 'attn':
                self.filter = TransformerEncoder(self.args)
        else:
            self.Controller = Controller(self.args)
        if self.args.model == 'DETECTOR':
            self.attn_encoder = args.detector          
# model.attn_encoder(data.X[:, :args.offset, :], is_train=False)
# np.take_along_axis(np.array(model.attn_encoder.attention_weights[:, 0, 1:]), np.reshape(data.noise_amount, (-1, 1)), axis=1).mean()  #0.15921913
            threshold_list = [th/100. for th in range(1, 21, 1)]
            self.mse_list, self.mae_list = [], []
            for thr in threshold_list:
                over_threshold = tf.where(self.attn_encoder.attention_weights[:, 0, 1:] > thr, 1, 0)
                estimated_tr = np.argmax(over_threshold, axis=1)
                self.mse_list.append(np.square(np.subtract(estimated_tr, self.args.noise_amount[self.args.test_idx])).mean())
                self.mae_list.append(np.abs(np.subtract(estimated_tr, self.args.noise_amount[self.args.test_idx])).mean())
            min_idx = np.argmin(self.mse_list)
            self.detector_threshold = threshold_list[min_idx]
        self.BaselineNetwork = BaselineNetwork(self.args)
        self.LSTM = layers.LSTMCell(self.args.nhid)
        self.initial_states = tf.zeros([self.args.batch_size, self.args.nhid])
        self.out = layers.Dense(self.args.nclasses, activation='softmax')  # Discriminator
    
    def entropy(self, p):
        id_p = np.where(p != 0)
        return -np.sum(p[id_p]*np.log(p[id_p]))
        
    def call(self, X, y_true, is_train=True, pred_at=-1, length=None, noise_amount=0, tr_points=None):
        # attn_hidden, self.attn_logits = self.attn_encoder(X[:, :self.args.offset , :], is_train)
        if length is not None:
            length = length.reshape((-1, 1))   
        if is_train: # Model chooses for itself during testing
            self.Controller._epsilon = self._epsilon # set explore/exploit trade-off
        else:
            self.Controller._epsilon = 0.0
        B, T, V = X.shape # Input shape (BATCH x TIMESTEPS x VARIABLES)     
        
        if self.args.model in ['EARLIEST', 'ADAPTIVE']:
            hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
            start_point = 0
            self.attention_weights = []
        elif self.args.model == 'ATTENTION':
            attn_hidden, self.filter_logits = self.attn_encoder(X[:, :self.args.offset , :], is_train)
            hidden = [attn_hidden, tf.identity(self.initial_states[:B, :])]
            start_point = self.args.offset
            self.attention_weights = self.attn_encoder.attention_weights
        elif self.args.model =='NONE':
            hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
            start_point = self.args.offset
            self.attention_weights = []
        elif self.args.model == 'DETECTOR':
            self.filter_logits = None
            hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
            self.init_hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
            start_point = 0
            # diff = model.attn_encoder.attention_weights[:, 0, 2:] - model.attn_encoder.attention_weights[:, 0, 1:-1]
            # tf.argmax(diff, axis=1) + 1
            self.attn_encoder(X[:, :self.args.offset, :], is_train=False)
            over_threshold = np.where(self.attn_encoder.attention_weights[:, 0, 1:] > self.detector_threshold, 1, 0)
            self.estimated_tr = np.argmax(over_threshold, axis=1).reshape((-1,1))
            self.estimated_tr = np.where(self.estimated_tr < length, self.estimated_tr, 0)
            self.attention_weights = self.attn_encoder.attention_weights

        halt_points = -tf.ones([B,1])
        filter_points = tf.ones([B,1]) * self.args.seq_len
        predictions = tf.zeros([B, self.args.nclasses])
        filter_flags = tf.zeros([B, 1])
        
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        baselines = [] # Predicted baselines
        raw_probs = []
        pred_y = []
        distribution = []
        tr_window_hidden = []
        
        for t in range(start_point, T):
            x = X[:,t,:]
            output, hidden = self.LSTM(x, states=hidden)
            
            if self.args.model == "DETECTOR":
                hidden1 = tf.where(self.estimated_tr > t , self.init_hidden[0], hidden[0])
                hidden2 = tf.where(self.estimated_tr > t , self.init_hidden[1], hidden[1])
                hidden = [hidden1, hidden2]
            
            # predict logits for all elements in the batch
            logits = self.out(output)     
            yhat_t = tf.argmax(logits, 1)
            yhat_t = tf.reshape(yhat_t, (-1, 1))
            
            # compute halting probability, sample an action, and baseline
            if self.args.test_t:
                t = self.t
            time = tf.ones([B,1]) * t
            c_in = tf.stop_gradient(tf.concat([output, time], axis=1))
            a_t, p_t, w_t, probs_t = self.Controller(c_in)
            b_t = self.BaselineNetwork(c_in)
          
            if self.args.model == "DETECTOR":
                a_t = tf.where(self.estimated_tr > t , 0, a_t)
            if self.args.full_seq:
                a_t *= 0
            
            predictions = tf.where((length-1 <= t) & (predictions == 0), logits, predictions)
            predictions = tf.where((a_t == 1) & (predictions == 0), logits, predictions)
            probs_t = tf.where(halt_points == -1, probs_t, -1)
            yhat_t = tf.where(halt_points == -1, yhat_t, -1)
            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = tf.where((halt_points == -1) & (length-1 <= t), length-1, halt_points)
            halt_points = tf.where((halt_points == -1) & (a_t == 1), t, halt_points)
            
            actions.append(a_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            baselines.append(b_t)
            raw_probs.append(probs_t)
            pred_y.append(yhat_t)
            distribution.append(logits)
            if self.args.model == "ADAPTIVE" and np.sum((halt_points == -1)) == 0 and t >= self.args.offset:  # If no negative values, every class has been halted
                break
            if self.args.model != "ADAPTIVE" and np.sum((halt_points == -1)) == 0:
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
        self.filter_flags = filter_flags.numpy()
        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = np.zeros_like(self.actions)
        for b in range(B):
            if self.args.model in ["EARLIEST"]:
                self.grad_mask[b, :(1 + int(halt_points[b, 0]))] = 1
            elif self.args.model in ["ATTENTION", "NONE"]:
                self.grad_mask[b, :(1 + int(halt_points[b, 0]) - self.args.offset)] = 1
            elif self.args.model in ["ADAPTIVE"]:
                halt_points = tf.where(halt_points < filter_points, halt_points, filter_points)
                self.grad_mask[b, :(1 + int(halt_points[b, 0]))] = 1
            elif self.args.model in ["DETECTOR"]:
                self.grad_mask[b, :(1 + int(halt_points[b, 0]))] = 1
                self.grad_mask[b, :int(self.estimated_tr[b, 0])] = 0
        if self.args.model in ["DETECTOR"]:
            # halt_points = halt_points - self.estimated_tr
            self.locations_det = tf.where(self.locations < self.args.offset, self.args.offset, self.locations)
            self.locations_det = tf.where(self.locations_det > length, length, self.locations_det)          
            self.locations_det = self.locations_det.numpy()
            self.locations = self.locations - self.estimated_tr
        return logits

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embedding_dim, num_heads):
        super(TokenAndPositionEmbedding, self).__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        # self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)
        if num_heads != 1:
            self.sensor_emb = tf.keras.layers.Dense(embedding_dim, activation="sigmoid")

    def call(self, x):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_emb(positions)
        if self.num_heads != 1:
            x = self.sensor_emb(x)
        # x = self.token_emb(x)
        return x + positions

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim # d_model
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value)
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

        # (batch_size, seq_len, embedding_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs = self.dense(concat_attention)
        return outputs, attention_weights

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, args, embedding_dim, num_heads, dff):
        super(TransformerBlock, self).__init__()
        self.args = args
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation="relu"),
             tf.keras.layers.Dense(embedding_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.args.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.args.dropout_rate)

    # def call(self, inputs, training):
    def call(self, inputs):
        attn_output, attention_weights = self.att(inputs) # 첫번째 서브층 : 멀티 헤드 어텐션
        attn_output = self.dropout1(attn_output, training=self.training)
        out1 = self.layernorm1(inputs + attn_output) # Add & Norm
        ffn_output = self.ffn(out1) # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=self.training)
        attn_weight = tf.squeeze(attention_weights)
        # self.attention_weights = attn_weight[:, -1, :]
        self.attention_weights = attn_weight
        # return self.layernorm2(out1 + ffn_output), attention_weights # Add & Norm
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        if self.args.hidden_as_input or self.args.num_heads != 1:
            self.embedding_dim = self.args.nhid  
        else:
            self.embedding_dim = self.args.N_FEATURES  
        self.num_heads = self.args.num_heads 
        self.dff = self.args.dff
        self.num_encoder = self.args.num_encoder
        self.max_len = self.args.offset + 1
        if self.args.train_filter:
            self.max_len += 1
            self.out = layers.Dense(self.args.offset, activation='softmax')
            # self.dropout2 = tf.keras.layers.Dropout(self.args.dropout_rate)
        self.embedding_layer = TokenAndPositionEmbedding(self.max_len, self.embedding_dim, self.num_heads)
        self.transformer_blocks = tf.keras.Sequential()
        for _ in range(self.num_encoder):
            self.transformer_blocks.add(TransformerBlock(self.args, self.embedding_dim, self.num_heads, self.dff))
        # self.transformer_block = TransformerBlock(self.args, self.embedding_dim, self.num_heads, self.dff)
        # self.transformer_block = {}
        # for i in range(self.num_encoder):
        #     self.transformer_block[i] = TransformerBlock(self.args, self.embedding_dim, self.num_heads, self.dff)
        self.hidden = layers.Dense(self.args.nhid, activation='tanh')
        self.dropout1 = tf.keras.layers.Dropout(self.args.dropout_rate)
        
            
        
    def call(self, X, is_train):
        class_token = np.zeros((X.shape[0], 1, self.args.N_FEATURES))
        if self.args.train_filter:
            train_token = np.zeros((X.shape[0], 1, self.args.N_FEATURES))
            X = np.concatenate((class_token, train_token, X), axis=1)
        else:
            X = np.concatenate((class_token, X), axis=1)
            
        for i in range(self.num_encoder):
            self.transformer_blocks.layers[i].training = is_train
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        self.attention_weights = self.transformer_blocks.layers[-1].attention_weights
        # for _ in range(self.num_encoder):
        #     X, self.attention_weights = self.transformer_block[i](X, is_train)
        # X, self.attention_weights = self.transformer_block(X, is_train)        
        # if self.args.train_filter:
        #     hidden_states = self.hidden(X[:,0,:])
        #     outputs = self.out(self.dropout2(hidden_states, training=is_train))
        #     # outputs = self.out(X[:,1:,:])
        # else:
        #     # hidden_states = self.hidden(X[:,-1,:])
        #     hidden_states = self.hidden(X[:,0,:])
        #     outputs = None
            # if self.args.utilize_tr:
            #     tr_points = np.reshape(tr_points, (-1, 1, 1)) #dtype=tf.int32
            #     tr_points = tf.convert_to_tensor(tr_points, dtype=tf.int32)
            #     X = tf.experimental.numpy.take_along_axis(X, tr_points, axis=1)
            #     X = tf.squeeze(X)
            #     hidden_states = self.hidden(X)
            #     outputs = None
            # else:
        # if self.args.drop_context:
        hidden_states = self.hidden(X[:,0,:])
        hidden_states = self.dropout1(hidden_states, training=is_train)
        if self.args.train_filter:
            outputs = self.out(X[:,1,:])
        else:
            outputs = None
        return hidden_states, outputs
