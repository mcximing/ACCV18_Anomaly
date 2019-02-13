# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 06:54:52 2018

@author: Cong
"""


import os
import pickle
import numpy as np
from scipy import interpolate
from hmm_dist import train, hmmdist_eval


def remove_identities(traj):
    # traj shape: (M,2)
    diff = np.diff(traj, axis=0)
    dupl = (np.sum(np.abs(diff), axis=1)==0)
    dupl = np.r_[dupl, False]
    if dupl.sum() < 1:
        return traj
    traj[dupl] += 0.1
    return remove_identities(traj)


def resample(trajs, t_len=12):
    # spline fitting
    # trajs: list of trajectories
    t = np.linspace(0, 1, num=t_len) # fixed-length 
    dataset = []
    for traj in trajs:
        traj = remove_identities(traj[:,:2].astype(np.float))
        tck,u = interpolate.splprep(traj.T, k=3, s=0) # traj:(M,2)
        out = interpolate.splev(t,tck) # a list saving two arrays
        dataset.append(np.c_[out[0], out[1]]) # (T,2)
    dataset = np.array(dataset) # (N,T,2)
    return dataset


def dm_hmmdist(trajs, modeldir = './detrac_hmms/'):
    trajs = resample(trajs, t_len=12)
    data = []
    for t in trajs:
        t[:,0] /= 960.
        t[:,1] /= 540.
        data.append(t)
    trajs = data
    traj_num = len(trajs)
    
    # compute distance matrix
    print('training...')
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    
    M = np.array([1,1,1,1,1,1])
    
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
    
    namelist = []
    with open('D:/Data/DETRAC/train-names.txt', 'r') as fp:
        for line in fp:
            namelist.append(line.strip())
    
    dataset_path = 'D:/Data/DETRAC/train-traj-data/'
    with open(dataset_path+'data.pkl', 'rb') as fp:
        data = pickle.load(fp)
    with open(dataset_path+'label.pkl', 'rb') as fp:
        label = pickle.load(fp)
    
    # compute
    dms = []
    for i in range(60):
        print('---', i)
        name = namelist[i]
        if (label[name]==1).sum() > 0:
            dm = dm_hmmdist(data[name])
            dms.append(dm)
    
    with open('detrac_dms_hmm.pkl', 'wb') as dmfp:
        pickle.dump(dms, dmfp)
    
