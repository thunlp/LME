# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('../../')
sys.path.append('../')
from create_prior_knowledge import create_prior
from gumbel_softmax import gumbel_softmax

def weight_variable(name, shape, pad=True):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    if pad == True:
        initial[0] = np.zeros(shape[1])
    initial = tf.constant_initializer(initial)
    return tf.get_variable(name=name, shape=shape, initializer=initial)

def attentive_sum(inputs,input_dim, hidden_dim):
    with tf.variable_scope("attention"):
        seq_length = len(inputs)
        W =  weight_variable('att_W', (input_dim,hidden_dim))
        U =  weight_variable('att_U', (hidden_dim,1))
        tf.get_variable_scope().reuse_variables()
        temp1 = [tf.nn.tanh(tf.matmul(inputs[i],W)) for i in range(seq_length)]
        temp2 = [tf.matmul(temp1[i],U) for i in range(seq_length)]
        pre_activations = tf.concat(1,temp2)
        attentions = tf.split(1, seq_length, tf.nn.softmax(pre_activations))
        weighted_inputs = [tf.mul(inputs[i],attentions[i]) for i in range(seq_length)]
        output = tf.add_n(weighted_inputs)
    return output, attentions


class Model:
    def __init__(
        self,type = "figer", encoder = "averaging",
        hier = False, feature = False, LE=None, vocab_size=0, lamb=0.0,
        LMout=None, hard=False):

        # Argument Checking
        assert(encoder in ["averaging", "lstm", "attentive"])
        assert(type in ["figer", "gillick"])
        self.type = type
        self.encoder = encoder
        self.hier = hier
        self.feature = feature
        self.LE = LE
        self.hard = hard

        # Hyperparameters
        self.lamb = lamb
        self.context_length = 10
        self.emb_dim = 300
        self.target_dim = 113 if type == "figer" else 89
        self.feature_size = 600000 if type == "figer" else 100000
        self.learning_rate = 0.001
        self.lstm_dim = 100
        self.LM_dim = 500
        self.att_dim  = 100 # dim of attention module
        self.feature_dim = 50 # dim of feature representation
        self.feature_input_dim = 70
        self.vocab_size = vocab_size
        if encoder == "averaging":
            self.rep_dim = self.emb_dim * 3
        else:
            self.rep_dim = self.lstm_dim * 2 + self.emb_dim
        if feature:
            self.rep_dim += self.feature_dim

        self.build_typing_part()
        self.build_LM_part()
        

        # Loss Function
        self.type_loss = tf.reduce_mean(self.type_loss)
        self.LM_loss_total = tf.reduce_mean(
            self.LM_loss)#[:, self.context_length:self.context_length+2])
        # !!! probable problems here
        self.pre_loss = tf.reduce_mean(self.LM_pre_loss)

        self.loss = self.type_loss# + self.lamb * self.LM_loss_total




        # Optimizer

        LM_namelist = ["LM_W:0",
                       "LM_b:0",
                       "LM/RNN/LSTMCell/W_0:0",
                       "LM/RNN/LSTMCell/B:0",
                       "LM2/RNN/LSTMCell/W_0:0",
                       "LM2/RNN/LSTMCell/B:0"]
        LM_exclude = [v for v in tf.all_variables() if v.name not in LM_namelist]
        self.optim = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = self.optim.minimize(self.loss, var_list=LM_exclude)
        self.pre_optim = tf.train.AdamOptimizer(self.learning_rate)
        self.pre_trainer = self.pre_optim.minimize(self.pre_loss)


        # Session
        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(self.init)

        # Saver
        LM_savelist = [v for v in tf.all_variables() if v.name in LM_namelist]
        self.LM_saver = tf.train.Saver(LM_savelist,
            max_to_keep=100, name="LMSaver")
        self.saver = tf.train.Saver(max_to_keep=100, name="Saver")

    def build_typing_part(self):
        # Place Holders
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.mention_representation = tf.placeholder(tf.float32, 
            [None,self.emb_dim], name="mention_repr")
        self.context = [
            tf.placeholder(tf.float32, [None, self.emb_dim], name="context"+str(i))
            for i in range(self.context_length*2+1)]
        self.target = tf.placeholder(tf.float32,
            [None,self.target_dim], name="target")
        ### dropout and splitting context into left and right
        self.mention_representation_dropout = tf.nn.dropout(self.mention_representation,self.keep_prob)
        self.left_context = self.context[:self.context_length]
        self.right_context = self.context[self.context_length+1:]


        
        # Averaging Encoder
        if self.encoder == "averaging":
            self.left_context_representation  = tf.add_n(self.left_context)
            self.right_context_representation = tf.add_n(self.right_context)
            self.context_representation       = tf.concat(1,[self.left_context_representation,self.right_context_representation])

        # LSTM Encoder
        if self.encoder == "lstm":
            self.left_lstm  = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.right_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            with tf.variable_scope("rnn_left") as scope:
                self.left_rnn,_  = tf.nn.rnn(self.left_lstm,self.left_context,dtype=tf.float32)
            with tf.variable_scope("rnn_right") as scope:
                self.right_rnn,_ = tf.nn.rnn(self.right_lstm,list(reversed(self.right_context)),dtype=tf.float32)
            self.context_representation = tf.concat(1,[self.left_rnn[-1],self.right_rnn[-1]])

        # Attentive Encoder
        if self.encoder == "attentive":
            self.left_lstm_F  = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.right_lstm_F = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.left_lstm_B  = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.right_lstm_B = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            with tf.variable_scope("rnn_left") as scope:
                self.left_birnn,_,_  = tf.nn.bidirectional_rnn(self.left_lstm_F,self.left_lstm_B,self.left_context,dtype=tf.float32)
            with tf.variable_scope("rnn_right") as scope:
                self.right_birnn,_,_ = tf.nn.bidirectional_rnn(self.right_lstm_F,self.right_lstm_B,list(reversed(self.right_context)),dtype=tf.float32)
            self.context_representation, self.attentions = attentive_sum(self.left_birnn + self.right_birnn, input_dim = self.lstm_dim * 2, hidden_dim = self.att_dim)


        # Logistic Regression
        if self.feature:
            self.features = tf.placeholder(tf.int32,[None,self.feature_input_dim])
            self.feature_embeddings = weight_variable('feat_embds', (self.feature_size, self.feature_dim), True)
            self.feature_representation = tf.nn.dropout(tf.reduce_sum(tf.nn.embedding_lookup(self.feature_embeddings,self.features),1),self.keep_prob)
            self.representation = tf.concat(1, [self.mention_representation_dropout, self.context_representation, self.feature_representation])
        else:
            self.representation = tf.concat(1, [self.mention_representation_dropout, self.context_representation])

        if self.hier:
            _d = "Wiki" if self.type == "figer" else "OntoNotes"
            S = create_prior("./resource/"+_d+"/label2id_"+self.type+".txt")
            assert(S.shape == (self.target_dim, self.target_dim))
            self.S = tf.constant(S,dtype=tf.float32)
            self.V = weight_variable('hier_V', (self.target_dim,self.rep_dim))
            self.W = tf.transpose(tf.matmul(self.S,self.V))
            self.logit = tf.matmul(self.representation, self.W)
        else:
            self.W = weight_variable('hier_W', (self.rep_dim,self.target_dim))
            self.logit = tf.matmul(self.representation, self.W)

        self.distribution = tf.nn.sigmoid(self.logit)
        self.type_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.logit, self.target)

    def build_LM_part(self):
        # Label Embedding (LE) and Language Model (LM)
        self.label_embedding = tf.Variable(self.LE, name="LE", trainable=False)
        if not self.hard:
            self.label_input = tf.matmul(
                tf.nn.softmax(self.logit), self.label_embedding)
        else:
            self.label_input = tf.matmul(
                gumbel_softmax(self.logit, 0.5, False), self.label_embedding)


        # Place Holders
        self.LM_keep_prob = tf.placeholder(tf.float32, name="LM_keep_prob")
        self.LM_pre_context = [
                tf.placeholder(tf.float32,
                [None, self.emb_dim], name="pre_context"+str(i))
            for i in range(self.context_length*2+1)]
        self.LM_context = [
                tf.placeholder(tf.float32,
                [None, self.emb_dim], name="context"+str(i))
            for i in range(self.context_length*2+1)]
        self.LM_sp_in = tf.sparse_placeholder(tf.int32, name="sp_in")
            # [2*self.context_length+1, None, 1], name="sp_in")
        self.context_id = tf.placeholder(tf.int32,
            [None, self.context_length*2+1], name="context_id")
        


        # LM parameters
        self.LM = tf.nn.rnn_cell.LSTMCell(self.LM_dim, state_is_tuple=True)
        self.LM2 = tf.nn.rnn_cell.LSTMCell(self.emb_dim, state_is_tuple=True)
        # self.LM_W = tf.Variable(tf.to_float(tf.transpose(LMout)),
        #     name="LM_W")
        self.LM_W = weight_variable(
            "LM_W", [self.emb_dim, self.vocab_size], pad=False)
        self.LM_b = weight_variable(
            "LM_b", [self.vocab_size], pad=False)


        # LM pretrain
        self.LM_pre_in = tf.pack(self.LM_pre_context)
        with tf.variable_scope("LM") as scope:
            self.LM_pre_out_, _ = tf.nn.dynamic_rnn(
                self.LM, self.LM_pre_in,
                time_major=True, dtype=tf.float32)
        self.LM_pre_out_ = tf.nn.dropout(self.LM_pre_out_, self.LM_keep_prob)
        with tf.variable_scope("LM2") as scope:
            self.LM_pre_out, _ = tf.nn.dynamic_rnn(
                self.LM2, self.LM_pre_out_,
                time_major=True, dtype=tf.float32)
        self.LM_pre_outs = tf.reshape(self.LM_pre_out, [-1, self.emb_dim])
        self.LM_pre_logit = tf.matmul(self.LM_pre_outs, self.LM_W) + self.LM_b
        self.LM_pre_logit = tf.reshape(self.LM_pre_logit,
            [-1, self.context_length*2+1, self.vocab_size])[:, :-1, :]
        self.LM_pre_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.LM_pre_logit, self.context_id[:, 1:])


        # LM in typing
        self.sp = tf.contrib.layers.safe_embedding_lookup_sparse(
                [self.label_input], self.LM_sp_in, combiner="sum")
        self.sp.set_shape([self.context_length*2+1, None, self.emb_dim])
        self.LM_in = tf.pack(self.LM_context) + self.sp #!!!!!!!!!!!!!!

        with tf.variable_scope("LM", reuse=True) as scope:
            self.LM_out_, _ = tf.nn.dynamic_rnn(
                self.LM, self.LM_in,
                time_major=True, dtype=tf.float32)
        self.LM_out_ = tf.nn.dropout(self.LM_out_, self.LM_keep_prob)
        with tf.variable_scope("LM2", reuse=True) as scope:
            self.LM_out, _ = tf.nn.dynamic_rnn(
                self.LM2, self.LM_out_,
                time_major=True, dtype=tf.float32)
        self.LM_outs = tf.reshape(self.LM_out, [-1, self.emb_dim])
        self.LM_logit = tf.matmul(self.LM_outs, self.LM_W) + self.LM_b
        self.LM_logit = tf.reshape(self.LM_logit,
            [-1, self.context_length*2+1, self.vocab_size])[:, :-1, :]
        self.LM_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.LM_logit, self.context_id[:, 1:])

    

    def _feed(self, batch, kp=1.0, lkp=1.0):
        feed = {
                self.keep_prob: kp,
                self.LM_keep_prob: lkp,
                self.mention_representation: batch["target_mean"],
                self.target: batch["Y"],
                self.context_id: batch["context_id"],
                }
        if self.feature == True and batch["F"] is not None:
            feed[self.features] = batch["F"]
        cmid = batch["cmid"]
        sp_in = [[], []]
        for i in range(self.context_length*2+1):
            feed[self.context[i]] = batch["context"][:,i,:]
            vec = batch["LM_context"][:, i, :]
            feed[self.LM_pre_context[i]] = [tuple(ent) for ent in vec]
            # otherwise it would be changed
            for j in range(len(cmid)): # i: timestep; j: batch_num
                if i==cmid[j]:
                    vec[j] = np.zeros([self.emb_dim])
                    sp_in[0].append([i, j, 0])
                    sp_in[1].append(j)
            feed[self.LM_context[i]] = vec
        sp_in.append([2*self.context_length+1, len(cmid), 1])
        feed[self.LM_sp_in] = sp_in

        return feed

    def printer(self, plist, batch, mute=False):
        feed = self._feed(batch)

        def pr(what):
            tmp = self.session.run(what, feed_dict=feed)
            if not mute:
                print(tmp)
                print(tmp.shape)
            return tmp


        ans = []
        for p in plist:
            bigtmp = []
            if isinstance(p, list):
                for pp in p:
                    tmp = pr(pp)
                    bigtmp.append(tmp)
            else:
                tmp = pr(p)
                bigtmp = tmp
            ans.append(bigtmp)
        if not mute:
            print
        return ans

    def train(self, batch):
        feed = self._feed(batch, kp=0.5)
        self.session.run(self.trainer,feed_dict=feed)

    def pretrain(self, batch):
        feed = self._feed(batch, lkp=0.5)
        self.session.run(self.pre_trainer,feed_dict=feed)

    def predict(self, batch):
        feed = self._feed(batch)
        return self.session.run(self.distribution,feed_dict=feed)

    def save_LM(self, step, id):
        print("Saving LM model")
        try:
            os.mkdir("./LMmodel/"+id)
        except OSError as e:
            if e.errno!=17: #File exists
                raise e
        self.LM_saver.save(self.session, "./LMmodel/"+id+"/model", step)


    def save_all(self, step, id):
        print("Saving the entire model")
        try:
            os.mkdir("./Emodel/"+id)
        except OSError as e:
            if e.errno!=17: #File exists
                raise e
        self.saver.save(self.session, "./Emodel/"+id+"/model", step)

    
    def load_LM(self, path=None):
        print("Loading LM model")
        if path is None:
            self.LM_saver.restore(self.session, "./LMmodel/debug-1000/model")
        else:
            self.LM_saver.restore(self.session, path)

    def load_all(self, path=None):
        print("Loading the model")
        if path is None:
            self.saver.restore(self.session,
                "./Models/Wiki/lamb0.005/model")
        else:
            self.saver.restore(self.session, path)
            print(path)

    def save_label_embeddings(self):
        pass
