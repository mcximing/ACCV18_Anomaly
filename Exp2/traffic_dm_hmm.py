# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 05:43:58 2018

@author: Cong
"""

import os
import pickle
import numpy as np
from hmm_dist import train, hmmdist_eval
from datasets import read_traffic_dataset


if __name__ == '__main__':
    
    # read dataset
    train_set, train_labels, test_set, true_labels = read_traffic_dataset(
            resample=True, read_name='traffic_set_idx_06290219.txt')
    train_num = len(train_labels)
    test_num = len(true_labels)
    
    M = np.array([1,1,1,1,1,1])
    
    print('training')
    modeldir = './traffic_hmms/'
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    
    for i in range(train_num):
        print('--',i)
        train_traj = train_set[i] # (tlen, 2)
        pos_embed = np.arange(0, len(train_traj), dtype=np.float).reshape(-1,1) / len(train_traj)
        train_traj = np.c_[train_traj, pos_embed]
        # train
        hmm = train([train_traj], M)
        with open(modeldir+'hmm_train_%d.pkl'%i, 'wb') as fp:
            pickle.dump(hmm, fp)
    for j in range(test_num):
        print('--',j)
        test_traj = test_set[j] # (tlen, 2)
        pos_embed = np.arange(0, len(test_traj), dtype=np.float).reshape(-1,1) / len(test_traj)
        test_traj = np.c_[test_traj, pos_embed]
        # train
        hmm = train([test_traj], M)
        with open(modeldir+'hmm_test_%d.pkl'%j, 'wb') as fp:
            pickle.dump(hmm, fp)
    
    # compute distances
    print('compute distances')
    dist_mat = np.zeros((train_num, test_num))
    for i in range(train_num):
        print('-', i)
        train_traj = train_set[i] # (tlen, 2)
        pos_embed = np.arange(0, len(train_traj), dtype=np.float).reshape(-1,1) / len(train_traj)
        train_traj = np.c_[train_traj, pos_embed]
        with open(modeldir+'hmm_train_%d.pkl'%i, 'rb') as fp:
            hmm_train = pickle.load(fp)
        
        for j in range(test_num):
            #print('-', j)
            test_traj = test_set[j]
            pos_embed_test = np.arange(0, len(test_traj), dtype=np.float).reshape(-1,1) / len(test_traj)
            test_traj = np.c_[test_traj, pos_embed_test]
            with open(modeldir+'hmm_test_%d.pkl'%j, 'rb') as fp:
                hmm_test = pickle.load(fp)
            
            dist_mat[i,j] = hmmdist_eval(hmm_train, hmm_test, train_traj, test_traj)
    
    np.savez_compressed('traffic_dm_hmm', dist_mat=dist_mat)
    