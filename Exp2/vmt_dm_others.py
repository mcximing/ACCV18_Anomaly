# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 06:36:16 2018

@author: Cong
"""

import numpy as np
import scipy.spatial.distance as sd
import traj_dist.distance as tdist

from datasets import read_VMT


if __name__ == "__main__":
    
    # read dataset
    trajs, labels = read_VMT()
    traj_num = len(trajs)

    y = tdist.pdist(trajs, metric='dtw')
    dm_dtw = sd.squareform(y)
    np.savez_compressed('vmt_dm_dtw', dm_dtw)
    
    y = tdist.pdist(trajs, metric='sspd')
    dm_sspd = sd.squareform(y)
    np.savez_compressed('vmt_dm_sspd', dm_sspd)
    
    y = tdist.pdist(trajs, metric='lcss', eps=0.05)
    dm_lcss = sd.squareform(y)
    np.savez_compressed('vmt_dm_lcss', dm_lcss)
    
    y = tdist.pdist(trajs, metric='edr', eps=0.05)
    dm_edr = sd.squareform(y)
    np.savez_compressed('vmt_dm_edr', dm_edr)
    
    y = tdist.pdist(trajs, metric='erp', g = np.zeros(2,dtype=float))
    dm_erp = sd.squareform(y)
    np.savez_compressed('vmt_dm_erp', dm_erp)
    
    y = tdist.pdist(trajs, metric='hausdorff')
    dm_hau = sd.squareform(y)
    np.savez_compressed('vmt_dm_hau', dm_hau)
    
    y = tdist.pdist(trajs, metric='frechet')
    dm_fre = sd.squareform(y)
    np.savez_compressed('vmt_dm_fre', dm_fre)
    