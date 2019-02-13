# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:36:29 2018

@author: Cong
"""

import pickle
import numpy as np
import scipy.spatial.distance as sd
import traj_dist.distance as tdist

from rae_dist import aeowd, aebid
from hmm_dist import hmmdist
from traj_patterns import Translation, Deviation, Opposite, Wait, Loop, Speed

from scipy import interpolate


def resample(traj, t_len=12):
    # spline fitting
    # trajs: list of trajectories
    t = np.linspace(0, 1, num=t_len) # fixed-length 
    tck,u = interpolate.splprep(traj.T, k=3, s=0) # traj:(M,2)
    out = interpolate.splev(t,tck) # a list saving two arrays
    traj = np.c_[out].T # (T,2)
    return traj


def eucdist(sample_a, sample_b):
    if len(sample_a) != len(sample_b):
        raise ValueError("Trajectory lengths must be same!")
    
    tlen = len(sample_a)
    dists = np.zeros((tlen,))
    for i in range(tlen):
        dists[i] = sd.euclidean(sample_a[i], sample_b[i])
    return dists.mean()


def compute_dists(trajs):
    pattern_num = len(trajs)
    sample_num = len(trajs[0])
    res_euc = np.zeros((pattern_num, sample_num))
    res_dtw = np.zeros((pattern_num, sample_num))
    res_sspd = np.zeros((pattern_num, sample_num))
    res_lcss = np.zeros((pattern_num, sample_num))
    res_edr = np.zeros((pattern_num, sample_num))
    res_erp = np.zeros((pattern_num, sample_num))
    res_fre = np.zeros((pattern_num, sample_num))
    res_hau = np.zeros((pattern_num, sample_num))
    res_hmm = np.zeros((pattern_num, sample_num))
    res_aes1 = np.zeros((pattern_num, sample_num))
    res_aes2 = np.zeros((pattern_num, sample_num))
    
    for i in range(pattern_num):
        print('-', i) # patterns
        for j in range(sample_num):
            print(i, '-', j) # samples
            sample1 = trajs[i][j][0]
            sample2 = trajs[i][j][1]
            if i > 2:
                norm_sample1 = resample(trajs[i][j][0])
                norm_sample2 = resample(trajs[i][j][1])
                res_euc[i,j] = eucdist(norm_sample1, norm_sample2)
            else:
                res_euc[i,j] = eucdist(sample1, sample2)
            
            res_dtw[i,j] = tdist.dtw(sample1, sample2)
            res_sspd[i,j] = tdist.sspd(sample1, sample2)
            res_lcss[i,j] = tdist.lcss(sample1, sample2, eps=0.6/tlen)
            res_edr[i,j] = tdist.edr(sample1, sample2, eps=0.6/tlen)
            res_erp[i,j] = tdist.erp(sample1, sample2, g = np.zeros(2,dtype=float))
            res_fre[i,j] = tdist.frechet(sample1, sample2)
            res_hau[i,j] = tdist.hausdorff(sample1, sample2)
            res_hmm[i,j] = hmmdist(sample1, sample2, M=np.array([1,1,1,1,1,1]))
            res_aes1[i,j] = aeowd(sample1, sample2, loss_th=1e-4)
            res_aes2[i,j] = aebid(sample1, sample2, loss_th=1e-4)
            
    results = np.array([res_euc, res_dtw, res_sspd, res_lcss, res_edr, res_erp, \
                        res_fre, res_hau, res_hmm, res_aes1, res_aes2]) # (12,8,11)
    return results
    

if __name__ == '__main__':
    
    tlen = 50
    pattern_num = 8
    sample_num = 10
    
    '''
    # generate trajectories
    print('generate trajectories')
    classes = [Translation, Deviation, Opposite, Loop, Wait, Speed]
    
    bases = [[] for _ in range(pattern_num)]
    trajs = [[] for _ in range(pattern_num)]
    for i in range(pattern_num):
        #print(i)
        for _ in range(sample_num):
            c = classes[i](tlen=tlen)
            bases[i].append([c.t0, c.t1])
            trajs[i].append([c.t0, c.t2])
    
    with open('bases.pkl', 'wb') as fp1:
        pickle.dump(bases, fp1)
    with open('trajs.pkl', 'wb') as fp2:
        pickle.dump(trajs, fp2)
    '''
    
    # compute distances
    print('compute distances')
    with open('bases.pkl', 'rb') as fp1:
        bases = pickle.load(fp1)
    with open('trajs.pkl', 'rb') as fp2:
        trajs = pickle.load(fp2)
    
    results1 = compute_dists(bases)
    results2 = compute_dists(trajs)
    #np.save('res_base', results1)
    #np.save('res_traj', results2)
    
    avg_res1 = np.mean(results1, axis=-1)
    avg_res2 = np.mean(results2, axis=-1)
    #np.save('comparison1', avg_res1)
    #np.save('comparison2', avg_res2)
    
    ratio = avg_res2 / (avg_res1 + np.finfo(np.float).tiny)
    np.save('ratio', ratio)
    