# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 05:35:05 2018

@author: Cong
"""


import numpy as np
import scipy.spatial.distance as sd
import traj_dist.distance as tdist

from datasets import read_traffic_dataset


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
    # read_name='traffic_set_idx_06210519.txt'
    # read_name='traffic_set_idx_06260152.txt'
    # read_name='traffic_set_idx_06260227.txt'
    # read_name='traffic_set_idx_06280622.txt'
    # read_name='traffic_set_idx_06290219.txt'
    train_set, train_labels, test_set, truth_labels = read_traffic_dataset(resample=True, randomize=True)
    train_num = len(train_labels)
    test_num = len(truth_labels)
    
    dm_eucl = np.zeros((train_num, test_num))
    for i in range(train_num):
        print(i)
        for j in range(test_num):
            dm_eucl[i,j] = eucdist(train_set[i], test_set[j])
    
    dm_dtw = tdist.cdist(train_set, test_set, metric='dtw')
    dm_sspd = tdist.cdist(train_set, test_set, metric='sspd')
    dm_lcss = tdist.cdist(train_set, test_set, metric='lcss', eps=0.05)
    dm_edr = tdist.cdist(train_set, test_set, metric='edr', eps=0.05)
    dm_erp = tdist.cdist(train_set, test_set, metric='erp', g = np.zeros(2,dtype=float))
    dm_fre = tdist.cdist(train_set, test_set, metric='frechet')
    dm_hau = tdist.cdist(train_set, test_set, metric='hausdorff')
    
    np.savez_compressed('traffic_dm_eucl', dm_eucl)
    np.savez_compressed('traffic_dm_dtw', dm_dtw)
    np.savez_compressed('traffic_dm_sspd', dm_sspd)
    np.savez_compressed('traffic_dm_lcss', dm_lcss)
    np.savez_compressed('traffic_dm_edr', dm_edr)
    np.savez_compressed('traffic_dm_erp', dm_erp)
    np.savez_compressed('traffic_dm_fre', dm_fre)
    np.savez_compressed('traffic_dm_hau', dm_hau)
    