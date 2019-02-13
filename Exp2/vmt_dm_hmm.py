# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:18:23 2018

@author: Cong
"""

import os
import pickle
import numpy as np
from datasets import read_VMT
from hmm_dist import train, hmmdist_eval


def dm_hmmdist(trajs, modeldir = './vmt_hmms/'):
    traj_num = len(trajs)
    
    # compute distance matrix
    print('training...')
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    
    M = np.array([1,1,1,1])
    
    for i in range(traj_num):
        print('--', i)
        traj = trajs[i]
        pos_embed = np.arange(0, len(traj), dtype=np.float).reshape(-1,1) / len(traj)
        traj = np.c_[traj, pos_embed]
        hmm = train([traj], M)
        with open(modeldir+'hmm%d.pkl'%i, 'wb') as fp:
            pickle.dump(hmm, fp)
    
    print('computing...')
    dm = np.zeros((traj_num, traj_num))
    for i in range(traj_num):
        print('-', i)
        source = trajs[i]
        pos_embed = np.arange(0, len(source), dtype=np.float).reshape(-1,1) / len(source)
        source = np.c_[source, pos_embed]
        
        with open(modeldir+'hmm%d.pkl'%i, 'rb') as fp:
            hmm_train = pickle.load(fp)
        
        for j in range(i+1, traj_num):
            if j%100==0:
                print(j)
            target = trajs[j]
            pos_embed2 = np.arange(0, len(target), dtype=np.float).reshape(-1,1) / len(target)
            target = np.c_[target, pos_embed2]
            with open(modeldir+'hmm%d.pkl'%j, 'rb') as fp:
                hmm_test = pickle.load(fp)
            dm[i,j] = hmmdist_eval(hmm_train, hmm_test, source, target)
    
    for i in range(traj_num):
        for j in range(traj_num):
            if i > j:
                dm[i,j] = dm[j,i]
    return dm



if __name__ == "__main__":
    
    # read dataset
    trajs, labels = read_VMT('/home/cong/Data/trj/CASIA_tjc.mat')
    dm = dm_hmmdist(trajs)
    np.savez_compressed('vmt_dm_hmm', dm)
