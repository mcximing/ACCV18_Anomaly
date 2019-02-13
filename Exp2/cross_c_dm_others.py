# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 04:52:49 2018

@author: Cong
"""

import numpy as np
import scipy.spatial.distance as sd
import traj_dist.distance as tdist
from datasets import read_cross_dataset
from scipy import interpolate


def resample_set(trajs, t_len=12):
    # spline fitting
    # trajs: list of trajectories
    t = np.linspace(0, 1, num=t_len) # fixed-length 
    dataset = []
    for traj in trajs:
        tck,u = interpolate.splprep(traj.T, k=3, s=0) # traj:(M,3)
        out = interpolate.splev(t,tck) # a list saving two arrays
        dataset.append(np.c_[out[0], out[1]]) # (T,2)
    dataset = np.array(dataset) # (N,T,2)
    return dataset


def eucdist(sample_a, sample_b):
    if len(sample_a) != len(sample_b):
        raise ValueError("Trajectory lengths must be same!")
    
    tlen = len(sample_a)
    dists = np.zeros((tlen,))
    for i in range(tlen):
        dists[i] = sd.euclidean(sample_a[i], sample_b[i])
    return dists.mean()


if __name__ == '__main__':
    
    # read dataset
    train_set, train_labels, test_set, true_labels = read_cross_dataset()
    train_num = len(train_labels)
    test_num = len(true_labels)
    c_num = int(train_labels.max())
    
    # compute distances
    dm_tst_eucl = np.zeros((c_num, test_num))
    dm_tst_dtw = np.zeros((c_num, test_num))
    dm_tst_sspd = np.zeros((c_num, test_num))
    dm_tst_lcss = np.zeros((c_num, test_num))
    dm_tst_edr = np.zeros((c_num, test_num))
    dm_tst_erp = np.zeros((c_num, test_num))
    dm_tst_fre = np.zeros((c_num, test_num))
    dm_tst_hau = np.zeros((c_num, test_num))
    
    for c in range(c_num):
        print('--', c)
        idx = np.arange(0, train_num, dtype=np.int)[train_labels==(c+1)]
        sample_set = np.array([train_set[i] for i in idx])
        train_traj = np.mean(sample_set, axis=0) # (tlen, 2)
        
        for j in range(test_num):
            if j%3000==0:
                print(j)
            test_traj = test_set[j]
            
            dm_tst_eucl[c,j] = eucdist(train_traj, test_traj)
            dm_tst_hau[c,j] = tdist.hausdorff(train_traj, test_traj)
            dm_tst_dtw[c,j] = tdist.dtw(train_traj, test_traj)
            dm_tst_sspd[c,j] = tdist.sspd(train_traj, test_traj)
            dm_tst_lcss[c,j] = tdist.lcss(train_traj, test_traj, eps=0.05)
            dm_tst_edr[c,j] = tdist.edr(train_traj, test_traj, eps=0.05)
            dm_tst_erp[c,j] = tdist.erp(train_traj, test_traj, g = np.zeros(2,dtype=float))
            dm_tst_fre[c,j] = tdist.frechet(train_traj, test_traj)
    
    np.savez_compressed('cross_c_dm_tst_eucl', dm_tst_eucl)
    np.savez_compressed('cross_c_dm_tst_dtw', dm_tst_dtw)
    np.savez_compressed('cross_c_dm_tst_sspd', dm_tst_sspd)
    np.savez_compressed('cross_c_dm_tst_lcss', dm_tst_lcss)
    np.savez_compressed('cross_c_dm_tst_edr', dm_tst_edr)
    np.savez_compressed('cross_c_dm_tst_erp', dm_tst_erp)
    np.savez_compressed('cross_c_dm_tst_fre', dm_tst_fre)
    np.savez_compressed('cross_c_dm_tst_hau', dm_tst_hau)
    
    