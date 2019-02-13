# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 08:29:25 2018

@author: Cong
"""

import os
import pickle
import numpy as np
from hmm_dist import train, hmmdist_eval
from datasets import read_cross_dataset


if __name__ == '__main__':
    
    # read dataset
    train_set, train_labels, test_set, true_labels = read_cross_dataset()
    train_num = len(train_labels)
    test_num = len(true_labels)
    c_num = int(train_labels.max())
    
    M = np.array([1,1,1,1,1,1])
    
    # train
    print('training')
    modeldir = './cross_hmms/'
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    
    # compute distances
    print('compute distances')
    dist_mat = np.zeros((c_num, test_num))
    for c in range(c_num):
        print('--', c)
        idx = np.arange(0, train_num, dtype=np.int)[train_labels==(c+1)]
        sample_set = np.array([train_set[i] for i in idx])
        train_traj = np.mean(sample_set, axis=0) # (tlen, 2)
        seq_len = len(train_traj)
        pos_embed = np.arange(0, seq_len, dtype=np.float).reshape(-1,1) / seq_len
        train_traj = np.c_[train_traj, pos_embed]
        
        # train
        hmm_train = train([train_traj], M)
    
        # evaluate
        for j in range(test_num):
            if j % 100 == 0:
                print(j)
            test_traj = test_set[j]
            pos_embed_test = np.arange(0, len(test_traj), dtype=np.float).reshape(-1,1) / len(test_traj)
            test_traj = np.c_[test_traj, pos_embed_test]
            
            with open(modeldir+'hmm_test_%d.pkl'%j, 'rb') as fp:
                hmm_test = pickle.load(fp)
            dist_mat[c,j] = hmmdist_eval(hmm_train, hmm_test, train_traj, test_traj)
    
    np.savez_compressed('cross_c_dm_hmm', dist_mat=dist_mat)
    