# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 07:01:06 2018

@author: Cong
"""


import numpy as np
import scipy.spatial.distance as sd

from datasets import read_VMT


def eucdist(sample_a, sample_b):
    if len(sample_a) != len(sample_b):
        raise ValueError("Trajectory lengths must be same!")
    
    tlen = len(sample_a)
    dists = np.zeros((tlen,))
    for i in range(tlen):
        dists[i] = sd.euclidean(sample_a[i], sample_b[i])
    return dists.mean()


if __name__ == "__main__":
    
    # read dataset
    trajs, labels = read_VMT()
    traj_num = len(trajs)
    
    dm = np.zeros((traj_num, traj_num))
    for i in range(traj_num):
        print('-', i)
        source = trajs[i]
        
        for j in range(i+1, traj_num):
            if j%100==0:
                print(j)
            target = trajs[j]
            dm[i,j] = eucdist(source, target)
    
    for i in range(traj_num):
        for j in range(traj_num):
            if i > j:
                dm[i,j] = dm[j,i]
    
    np.savez_compressed('vmt_dm_eucl', dm)
    