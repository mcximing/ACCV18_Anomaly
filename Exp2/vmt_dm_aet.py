# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:57:28 2018

@author: Cong
"""

import numpy as np
from datasets import read_VMT
from ptaet import aepdist


if __name__ == "__main__":
    
    # read dataset
    trajs, labels = read_VMT('/home/cong/Data/trj/CASIA_tjc.mat')
    dm = aepdist(trajs, modeldir='./vmt_statedicts/')
    np.savez_compressed('vmt_dm_aet', dm)