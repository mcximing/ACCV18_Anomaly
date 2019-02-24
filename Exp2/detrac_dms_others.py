# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 07:38:58 2018

@author: Cong
"""

import pickle
import numpy as np
import scipy.spatial.distance as sd
import traj_dist.distance as tdist
from scipy import interpolate


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
    
    namelist = sorted(list(labels.keys()))
    
    # compute
    dms_euc = []
    dms_dtw = []
    dms_sspd = []
    dms_lcss = []
    dms_edr = []
    dms_erp = []
    dms_fre = []
    dms_hau = []
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
            traj_num = len(trajs)
        
            y = sd.pdist(trajs.transpose(0,2,1).reshape(traj_num,-1), 'euclidean')
            dm_euc = sd.squareform(y)
            y = tdist.pdist(trajs, metric='dtw')
            dm_dtw = sd.squareform(y)
            y = tdist.pdist(trajs, metric='sspd')
            dm_sspd = sd.squareform(y)
            y = tdist.pdist(trajs, metric='lcss', eps=0.05)
            dm_lcss = sd.squareform(y)
            y = tdist.pdist(trajs, metric='edr', eps=0.05)
            dm_edr = sd.squareform(y)
            y = tdist.pdist(trajs, metric='erp', g = np.zeros(2,dtype=float))
            dm_erp = sd.squareform(y)
            y = tdist.pdist(trajs, metric='frechet')
            dm_fre = sd.squareform(y)
            y = tdist.pdist(trajs, metric='hausdorff')
            dm_hau = sd.squareform(y)
            
            dms_euc.append(dm_euc)
            dms_dtw.append(dm_dtw)
            dms_sspd.append(dm_sspd)
            dms_lcss.append(dm_lcss)
            dms_edr.append(dm_edr)
            dms_erp.append(dm_erp)
            dms_fre.append(dm_fre)
            dms_hau.append(dm_hau)
    
    with open('detrac_dms_euc.pkl', 'wb') as dmfp:
        pickle.dump(dms_euc, dmfp)
    with open('detrac_dms_dtw.pkl', 'wb') as dmfp:
        pickle.dump(dms_dtw, dmfp)
    with open('detrac_dms_sspd.pkl', 'wb') as dmfp:
        pickle.dump(dms_sspd, dmfp)
    with open('detrac_dms_lcss.pkl', 'wb') as dmfp:
        pickle.dump(dms_lcss, dmfp)
    with open('detrac_dms_edr.pkl', 'wb') as dmfp:
        pickle.dump(dms_edr, dmfp)
    with open('detrac_dms_erp.pkl', 'wb') as dmfp:
        pickle.dump(dms_erp, dmfp)
    with open('detrac_dms_fre.pkl', 'wb') as dmfp:
        pickle.dump(dms_fre, dmfp)
    with open('detrac_dms_hau.pkl', 'wb') as dmfp:
        pickle.dump(dms_hau, dmfp)
    

