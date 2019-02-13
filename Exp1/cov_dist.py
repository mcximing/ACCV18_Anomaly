# -*- coding: utf-8 -*-
"""
Implementing distance definition in the paper: 
He Ergezer, et al. Anomaly Detection and Activity Perception Using 
Covariance Descriptor for Trajectories. ECCVw 2016.
@author: mcxim
"""


import pickle
import numpy as np

def compute_covariance_descriptor(traj):
    # traj shape: (T,2)
    x, y = traj.T
    u = x[1:] - x[:-1]
    v = y[1:] - y[:-1]
    t = np.arange(1, len(traj), dtype=np.float)
    f = np.c_[x[1:], y[1:], u, v, t]
    c = np.cov(f.T) / (len(traj) - 1)
    e, q = np.linalg.eig(c)
    ed = np.diag( np.log(e) )
    log_c = q.dot(ed).dot(q.T)
    return log_c


def covdist(traj1, traj2):
    cd1 = compute_covariance_descriptor(traj1)
    cd2 = compute_covariance_descriptor(traj2)
    return np.linalg.norm(cd1 - cd2)


def compute_dists(trajs):
    pattern_num = len(trajs)
    sample_num = len(trajs[0])
    res_cov = np.zeros((pattern_num, sample_num))
    
    for i in range(pattern_num):
        print('-', i) # patterns
        for j in range(sample_num):
            print(i, '-', j) # samples
            sample1 = trajs[i][j][0]
            sample2 = trajs[i][j][1]
            res_cov[i,j] = covdist(sample1, sample2)
            
    results = np.array([res_cov]) # (1,8,10)
    return results



if __name__ == '__main__':
    
    tlen = 50
    pattern_num = 8
    sample_num = 10

    with open('bases.pkl', 'rb') as fp1:
        bases = pickle.load(fp1)
    with open('trajs.pkl', 'rb') as fp2:
        trajs = pickle.load(fp2)
        
    res1 = compute_dists(bases)
    res2 = compute_dists(trajs)
    
    avg_res1 = np.mean(res1, axis=-1)
    avg_res2 = np.mean(res2, axis=-1)
    
    ratio = avg_res2 / (avg_res1 + np.finfo(np.float).tiny)
    