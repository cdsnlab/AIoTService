# from this import d
from tkinter.tix import X_REGION
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

class Controller_Proposed(tf.keras.layers.Layer):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, args):
        super(Controller_Proposed, self).__init__()
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
        if self.args.model == 'CNN':
            self.cnn = CNNLayer(self.args)
        if self.args.model == 'PROPOSED':
            self.Controller = Controller_Proposed(self.args)
            if self.args.filter_name == 'attn':
                self.filter = TransformerEncoder(self.args)
            else:
                self.filter = CNNLayer(self.args)
        else:
            self.Controller = Controller(self.args)
        self.BaselineNetwork = BaselineNetwork(self.args)
        self.LSTM = layers.LSTMCell(self.args.nhid)
        self.initial_states = tf.zeros([self.args.batch_size, self.args.nhid])
        self.out = layers.Dense(self.args.nclasses, activation='softmax')  # Discriminator
    
    def entropy(self, p):
        id_p = np.where(p != 0)
        return -np.sum(p[id_p]*np.log(p[id_p]))
        
    def call(self, X, y_true, is_train=True, pred_at=-1, length=None, noise_amount=0, tr_points=None):
        # attn_hidden, self.attn_logits = self.attn_encoder(X[:, :self.args.offset , :], is_train)
                
        if is_train: # Model chooses for itself during testing
            self.Controller._epsilon = self._epsilon # set explore/exploit trade-off
        else:
            self.Controller._epsilon = 0.0
        B, T, V = X.shape # Input shape (BATCH x TIMESTEPS x VARIABLES)     
        
        if self.args.model in ['EARLIEST', 'PROPOSED']:
            hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
            start_point = 0
            self.attention_weights = []
        elif self.args.model == 'ATTENTION':
            attn_hidden, self.filter_logits = self.attn_encoder(X[:, :self.args.offset , :], is_train)
            hidden = [attn_hidden, tf.identity(self.initial_states[:B, :])]
            start_point = self.args.offset
            self.attention_weights = self.attn_encoder.attention_weights
        elif self.args.model == 'CNN':
            cnn_hidden, self.filter_logits = self.cnn(X[:, :self.args.offset , :], is_train)
            hidden = [cnn_hidden, tf.identity(self.initial_states[:B, :])]
            start_point = self.args.offset
            self.attention_weights = []
        elif self.args.model =='NONE':
            hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
            start_point = self.args.offset
            self.attention_weights = []

            
        halt_points = -tf.ones([B,1])
        filter_points = tf.ones([B,1]) * self.args.seq_len
        predictions = tf.zeros([B, self.args.nclasses])
        filter_flags = tf.zeros([B, 1])
        if length is not None:
            length = length.reshape((-1, 1))
        
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        baselines = [] # Predicted baselines
        raw_probs = []
        pred_y = []
        distribution = []
        tr_window_hidden = []
        
        # if not is_train and pred_at != -1:
        #     halt_points = pred_at / 100 * length + noise_amount
        #     for t in range(T):
        #         x = X[:,t,:]
        #         output, hidden = self.LSTM(x, states=hidden)
        #         # predict logits for all elements in the batch
        #         logits = self.out(output)
        #         predictions = tf.where((t > halt_points) & (predictions == 0), logits, predictions)
        #         if t > np.max(halt_points):
        #             break
        #     logits = tf.where(predictions == 0.0, logits, predictions)
        #     self.locations = halt_points.round().astype(int) - noise_amount
        #     return logits
        
        for t in range(start_point, T):
            x = X[:,t,:]
            if t == self.args.offset and self.args.model == "PROPOSED":
                if self.args.hidden_as_input:
                    tr_window_hidden = tf.concat(tr_window_hidden, axis=1)
                    filter_hidden, self.filter_logits = self.filter(tr_window_hidden, is_train)
                else:
                    filter_hidden, self.filter_logits = self.filter(X[:, :self.args.offset , :], is_train)
                if self.args.filter_name == 'attn':
                    self.attention_weights = self.filter.attention_weights
                else:
                    self.attention_weights = []
                # filter_flags = tf.where((filter_flags == 0) & (halt_points == -1), 1, filter_flags)
                hidden = tf.where(filter_flags == 1, filter_hidden, hidden)
            output, hidden = self.LSTM(x, states=hidden)
            
            # predict logits for all elements in the batch
            logits = self.out(output)     
            ent = -np.sum(logits*np.log(logits), axis=1).reshape(B, -1)
            yhat_t = tf.argmax(logits, 1)
            yhat_t = tf.reshape(yhat_t, (-1, 1))
            
            # compute halting probability, sample an action, and baseline
            if self.args.test_t:
                t = self.t
            time = tf.ones([B,1]) * t
            if self.args.entropy_halting:
                c_in = tf.stop_gradient(tf.concat([logits, ent, time], axis=1))
            else:
                c_in = tf.stop_gradient(tf.concat([output, time], axis=1))
            a_t, p_t, w_t, probs_t = self.Controller(c_in)
            b_t = self.BaselineNetwork(c_in)
            # if self.args.delay_halt and not is_train:
            # if self.args.delay_halt:
            #     cls_4 = np.array([0, 1, 2, 3])
            #     y_true = np.reshape(y_true, (B, -1))
            #     target_class = np.where(y_true == cls_4, 1, 0).sum(axis=1).reshape(B, -1)
            #     a_t = tf.where((ent > self.args.entropy_threshold) & (target_class == 1), 0, a_t)
            
            # if t < self.args.offset and self.args.read_all_tw:
            #     a_t = a_t * 0
            
            if t < self.args.offset and self.args.model == "PROPOSED":
                filter_points = tf.where((a_t == 2) & (filter_flags == 0), t, filter_points)
                filter_flags = tf.where((a_t == 2) & (halt_points == -1), 1, filter_flags)
                predictions = tf.where((length-1 <= t) & (predictions == 0) & (filter_flags == 0), logits, predictions)
                predictions = tf.where((a_t == 1) & (predictions == 0) & (filter_flags == 0), logits, predictions)
                probs_t = tf.where(halt_points == -1, probs_t, -1)
                yhat_t = tf.where(halt_points == -1, yhat_t, -1)
                # If a_t == 1 and this class hasn't been halted, save the time
                halt_points = tf.where((halt_points == -1) & (length-1 <= t) & (filter_flags == 0), length-1, halt_points)
                halt_points = tf.where((halt_points == -1) & (a_t == 1) & (filter_flags == 0), t, halt_points)
                tr_window_hidden.append(tf.reshape(output, [B, 1, -1]))
            else:
                # If a_t == 1 and this class hasn't been halted, save its logits
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
            if self.args.model == "PROPOSED" and np.sum((halt_points == -1)) == 0 and t >= self.args.offset:  # If no negative values, every class has been halted
                break
            if self.args.model != "PROPOSED" and np.sum((halt_points == -1)) == 0:
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
            elif self.args.model in ["ATTENTION", "CNN", "NONE"]:
                self.grad_mask[b, :(1 + int(halt_points[b, 0]) - self.args.offset)] = 1
            elif self.args.model in ["PROPOSED"]:
                halt_points = tf.where(halt_points < filter_points, halt_points, filter_points)
                self.grad_mask[b, :(1 + int(halt_points[b, 0]))] = 1
        return logits

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.max_len = max_len
        # self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_emb(positions)
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
        if self.args.hidden_as_input:
            self.embedding_dim = self.args.nhid  
        else:
            self.embedding_dim = self.args.N_FEATURES  
        self.num_heads = 1 
        self.dff = self.args.dff
        self.num_encoder = self.args.num_encoder
        self.max_len = self.args.offset + 1
        self.embedding_layer = TokenAndPositionEmbedding(self.max_len, self.embedding_dim)
        self.transformer_blocks = tf.keras.Sequential()
        for _ in range(self.num_encoder):
            self.transformer_blocks.add(TransformerBlock(self.args, self.embedding_dim, self.num_heads, self.dff))
        # self.transformer_block = TransformerBlock(self.args, self.embedding_dim, self.num_heads, self.dff)
        # self.transformer_block = {}
        # for i in range(self.num_encoder):
        #     self.transformer_block[i] = TransformerBlock(self.args, self.embedding_dim, self.num_heads, self.dff)
        self.hidden = layers.Dense(self.args.nhid, activation='tanh')
        self.out = layers.Dense(self.args.nclasses, activation='softmax')
        self.dropout1 = tf.keras.layers.Dropout(self.args.dropout_rate)
        
    def call(self, X, is_train):
        class_token = np.zeros((X.shape[0], 1, self.embedding_dim))
        X = np.concatenate((class_token, X), axis=1)
        for i in range(self.num_encoder):
            self.transformer_blocks.layers[i].training = is_train
        X = self.embedding_layer(X)
        X = self.transformer_blocks(X)
        self.attention_weights = self.transformer_blocks.layers[-1].attention_weights
        # for _ in range(self.num_encoder):
        #     X, self.attention_weights = self.transformer_block[i](X, is_train)
        # X, self.attention_weights = self.transformer_block(X, is_train)
        if self.args.train_filter:
            hidden_states = self.hidden(X[:,0,:])
            outputs = self.out(X[:,0,:])
        else:
            # hidden_states = self.hidden(X[:,-1,:])
            hidden_states = self.hidden(X[:,0,:])
            outputs = None
            # if self.args.utilize_tr:
            #     tr_points = np.reshape(tr_points, (-1, 1, 1)) #dtype=tf.int32
            #     tr_points = tf.convert_to_tensor(tr_points, dtype=tf.int32)
            #     X = tf.experimental.numpy.take_along_axis(X, tr_points, axis=1)
            #     X = tf.squeeze(X)
            #     hidden_states = self.hidden(X)
            #     outputs = None
            # else:
        # if self.args.drop_context:
        hidden_states = self.dropout1(hidden_states, training=is_train)
        return hidden_states, outputs

class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, args):
        super(CNNLayer, self).__init__()
        self.args = args
        
    def build(self, input_shape):
        self.cnn = tf.keras.Sequential([   
                    # tf.keras.layers.Conv1D(filters=self.args.filters, kernel_size=self.args.kernel_size, activation="relu", input_shape=(self.args.batch_size, self.args.offset, self.args.N_FEATURES)),
                    tf.keras.layers.Conv1D(filters=self.args.filters, kernel_size=self.args.kernel_size, activation="relu"),
                    tf.keras.layers.Conv1D(filters=self.args.filters, kernel_size=self.args.kernel_size, activation="relu"),
                    tf.keras.layers.Conv1D(filters=self.args.filters, kernel_size=self.args.kernel_size, activation="relu"),
                    tf.keras.layers.Flatten()
                    ])
        self.dropout1 = tf.keras.layers.Dropout(self.args.dropout_rate)
        self.hidden = layers.Dense(self.args.nhid, activation='tanh')
        self.out = layers.Dense(self.args.nclasses, activation='softmax')

    def call(self, x, is_train=True):
        x = self.cnn(x)
        x = self.dropout1(x, training=is_train)
        hidden_states = self.hidden(x)
        
        if self.args.train_filter:
            outputs = self.out(x)
        else:
            outputs = None
        return hidden_states, outputs


# input = data_natural.X[:4, :args.offset, :]
# input_shape = input.shape

# cnn = tf.keras.Sequential([   
#                 tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu", input_shape=(self.args.batch_size, self.args.offset, self.args.N_FEATURES)),
#                 tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu"),
#                 tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu"),
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dropout(self.args.dropout_rate)
#             ])
# a = cnn(input)
# a.shape



# class SEGMENTATION(tf.keras.Model):
#     def __init__(self, args):
#         super(SEGMENTATION, self).__init__(name='')
#         self.args = args
#         self.LSTM = layers.LSTMCell(self.args.nhid)
#         self.initial_states = tf.zeros([self.args.batch_size, self.args.nhid])
#         self.out = layers.Dense(1, activation='sigmoid')  # Discriminator
        
#     def call(self, X, tr_boundary, tr_point, lengths, is_train=True):
#         B, T, V = X.shape # Input shape (BATCH x TIMESTEPS x VARIABLES)     
#         hidden = [tf.identity(self.initial_states[:B, :]), tf.identity(self.initial_states[:B, :])]
#         # tr_predictions = -tf.ones([B, 1])
#         # pred_tr_points = -tf.ones([B, 1])
#         end_points = -tf.ones([B,1])
        
#         pred_tr_list = []
#         for t in range(T):
#             x = X[:,t,:]
#             output, hidden = self.LSTM(x, states=hidden)
#             pred_tr = self.out(output)
#             pred_tr_list.append(pred_tr)
            
#             end_points = tf.where(((tr_point + lengths) < t), 1, end_points)
#             if np.sum((end_points == -1)) == 0:  # If no negative values, every class has been halted
#                 break           
#         pred_tr = tf.concat(pred_tr_list, axis=1)
        
#         if is_train:
#             offset = 1
#             idx_pos_samples = [[i for i in range(p-offset, p+offset+1, 1) if i >= 0] for p in tr_point]
#             idx_neg_samples = []
#             for i, idx_pos_sam in enumerate(idx_pos_samples):
#                 range_pool = range(lengths[i] + tr_point[i]) if len(range(lengths[i] + tr_point[i])) < self.args.seq_len * 2 else range(self.args.seq_len * 2)
#                 pool = set(range_pool) - set(idx_pos_sam)
#                 replace = False if (len(pool) >= offset*2) else True
#                 idx_neg_samples.append(list(np.random.choice(list(pool), offset*2, replace=replace)))
#             idx_pred = [a+b for a,b in zip(idx_pos_samples, idx_neg_samples)]
#             idx_pred = tf.convert_to_tensor(idx_pred, dtype=tf.int32)
#             pred_tr = tf.experimental.numpy.take_along_axis(pred_tr, idx_pred, axis=1)
#             true_tr = tf.experimental.numpy.take_along_axis(tr_boundary, idx_pred, axis=1)
#             return pred_tr, true_tr
#         else:
#             return pred_tr, tr_boundary
        
        
        
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

# model = tf.keras.Sequential()
# model.add(TransformerBlock(args, 31, 1, 32))
# model.layers[0].args
# model.layers[0].training = 0

# tf.keras.backend.learning_phase()
# tf.keras.backend.set_learning_phase(1)
