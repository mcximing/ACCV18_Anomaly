# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:13:17 2018

@author: Cong
"""

import pickle
import numpy as np
import scipy.spatial.distance as sd
from scipy import interpolate


def eucdist(sample_a, sample_b):
    if len(sample_a) != len(sample_b):
        raise ValueError("Trajectory lengths must be same!")
    
    tlen = len(sample_a)
    dists = np.zeros((tlen,))
    for i in range(tlen):
        dists[i] = sd.euclidean(sample_a[i], sample_b[i])
    return dists.mean()


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


if __name__ == "__main__":
    
    dataset_path = 'D:/Data/DETRAC/train-traj-data/'
    with open(dataset_path+'data.pkl', 'rb') as fp:
        data = pickle.load(fp)
    with open(dataset_path+'label.pkl', 'rb') as fp:
        label = pickle.load(fp)
        
    namelist = sorted(list(label.keys()))
    
    # compute
    dms_eucl = []
    for s in range(60):
        print('--', s)
        name = namelist[s]
        if (label[name]==1).sum() > 0:
            trajs = resample(data[name])
            trajs_norm = []
            for t in trajs:
                t[:,0] /= 960.
                t[:,1] /= 540.
                trajs_norm.append(t)
            trajs = np.array(trajs_norm)
            traj_num = len(trajs)
            
            dm = np.zeros((traj_num, traj_num))
            for i in range(traj_num):
                for j in range(i+1, traj_num):
                    dm[i,j] = eucdist(trajs[i], trajs[j])
            
            for i in range(traj_num):
                for j in range(traj_num):
                    if i > j:
                        dm[i,j] = dm[j,i]
            dms_eucl.append(dm)
            
    with open('detrac_dms_eucl.pkl', 'wb') as dmfp:
        pickle.dump(dms_eucl, dmfp)
