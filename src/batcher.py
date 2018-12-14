import numpy as np
from sklearn.externals import joblib
import random


class Batcher:
    def __init__(self,storage,data,batch_size,context_length,id2vec,vocab_size):
        self.context_length = context_length
        self.storage = storage
        self.data = data
        self.num_of_samples = int(data.shape[0])
        self.dim = 300 #len(id2vec[0])
        self.num_of_labels = data.shape[1] - 4  - 70 
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(self.num_of_samples / self.batch_size) 
        self.id2vec = id2vec
        self.pad  = np.zeros(self.dim)
        self.pad[0] = 1.0
        self.vocab_size = vocab_size
        
    
    def create_input_output(self,row):
        s_start = row[0]
        s_end = row[1]
        e_start = row[2]
        e_end = row[3]
        labels = row[74:]
        features = row[4:74]
        seq_context = np.zeros((self.context_length*2 + 1,self.dim))
        LM_context = np.zeros((self.context_length*2 + 1,self.dim)) # np.repeat
        temp = [ self.id2vec[self.storage[i]] for i in range(e_start,e_end)]
        mean_target = np.mean(temp,axis=0)
        seq_context_id = [(self.vocab_size-1)
            for _ in range(self.context_length*2+1)]
        
        j = max(0,self.context_length - (e_start - s_start))
        k = 0
        # fix context_id here to combine two vocab_sizes
        for i in range(max(s_start,e_start - self.context_length),e_start):
            vec = self.id2vec[self.storage[i]]
            seq_context[j,:] = vec
            LM_context[k, :] = vec
            seq_context_id[k] = min(self.storage[i], self.vocab_size-1)
            j += 1
            k += 1
        seq_context[j,:] = np.ones_like(self.pad)
        LM_context[k, :] = mean_target
        seq_context_id[k] = min(self.storage[e_start], self.vocab_size-1)
        mid = k
        j += 1
        k += 1
        for i in range(e_end,min(e_end+self.context_length,s_end)):
            vec = self.id2vec[self.storage[i]]
            seq_context[j,:] = vec
            LM_context[k, :] = vec
            seq_context_id[k] = min(self.storage[i], self.vocab_size-1)
            j += 1
            k += 1


        return (seq_context, mean_target, labels,
            features, seq_context_id, LM_context, k, mid)
        

    def next(self):
        X_context = np.zeros((self.batch_size,self.context_length*2+1,self.dim))
        X_LM_context = np.zeros((self.batch_size,self.context_length*2+1,self.dim))
        X_context_id = np.zeros([self.batch_size, self.context_length*2+1], dtype=int)
        X_clen = np.zeros([self.batch_size], dtype=int)
        X_cmid = np.zeros([self.batch_size], dtype=int)
        X_target_mean = np.zeros((self.batch_size,self.dim)) 
        Y = np.zeros((self.batch_size,self.num_of_labels))
        F = np.zeros((self.batch_size,70),np.int32)
        for i in range(self.batch_size):
            (X_context[i,:,:], X_target_mean[i,:], Y[i,:], F[i,:],
                X_context_id[i, :], X_LM_context[i, :, :],
                X_clen[i], X_cmid[i]) = self.create_input_output(
                self.data[self.batch_num * self.batch_size + i,:])

        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return {
            "context": X_context,
            "target_mean": X_target_mean,
            "Y": Y,
            "F": F,
            "context_id": X_context_id,
            "LM_context": X_LM_context,
            "clen": X_clen,
            "cmid": X_cmid,
        }
                                        
    def shuffle(self):
        np.random.shuffle(self.data)



    

    
