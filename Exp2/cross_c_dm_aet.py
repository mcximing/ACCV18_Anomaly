# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:24:49 2018

@author: Cong
"""


import numpy as np
from ptaet import aecdist
from datasets import read_cross_dataset


if __name__ == '__main__':
    
    # read dataset
    train_set, train_labels, test_set, true_labels = read_cross_dataset()
    train_num = len(train_labels)
    test_num = len(true_labels)
    
    c_num = int(train_labels.max())
    center_set = []
    
    for c in range(c_num):
        idx = np.arange(0, train_num, dtype=np.int)[train_labels==(c+1)]
        sample_set = np.array([train_set[i] for i in idx])
        train_traj = np.mean(sample_set, axis=0) # (tlen, 2)
        center_set.append(train_traj)
    
    dist_mat = aecdist(center_set, test_set)
    
    np.savez_compressed('cross_c_dm_aet', dist_mat=dist_mat)
    