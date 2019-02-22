# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:59:06 2018

@author: Cong
"""

import pickle
import numpy as np
from scipy import interpolate
from ptaet import aepdist


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
    
    dataset_path = '/home/cong/Data/Anomaly/DETRAC/'
    with open(dataset_path+'data.pkl', 'rb') as fp:
        data = pickle.load(fp)
    with open(dataset_path+'label.pkl', 'rb') as fp:
        label = pickle.load(fp)
    
    namelist = list(label.keys())
    
    # compute
    dms = []
    for i in range(60):
        print('--', i)
        name = namelist[i]
        if (label[name]==1).sum() > 0:
            trajs = resample(data[name])
            trajs_norm = []
            for t in trajs:
                t[:,0] /= 960.
                t[:,1] /= 540.
                trajs_norm.append(t)
            trajs = np.array(trajs_norm)
            
            dm = aepdist(trajs, modeldir='./detrac_statedicts/')
            dms.append(dm)
    
    with open('detrac_dms_aet.pkl', 'wb') as dmfp:
        pickle.dump(dms, dmfp)
        
