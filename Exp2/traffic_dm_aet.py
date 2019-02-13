# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 06:08:15 2018

@author: Cong
"""


import numpy as np

from ptaet import aecdist
from datasets import read_traffic_dataset

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    
    # read dataset
    train_set, train_labels, test_set, true_labels = read_traffic_dataset(
            resample=True, read_name='traffic_set_idx_06290219.txt')
    train_num = len(train_labels)
    test_num = len(true_labels)
    
    # compute distance matrix
    dist_mat = aecdist(train_set, test_set)
    np.savez_compressed('traffic_dm_aet', dist_mat=dist_mat)
    
    # scores
    scores_test = np.sort(dist_mat, axis=0)[0]
    th = 0.0001
    
    # detection and evaluation
    gt_idx = np.arange(50, test_num, dtype=np.uint)
    gt = np.zeros((test_num,))
    gt[gt_idx] = 1
    
    det_idx = np.arange(0, test_num, dtype=np.uint)[scores_test>th]
    det = np.zeros((test_num,))
    det[det_idx] = 1
    
    cm = confusion_matrix(gt, det)
    tn, fp, fn, tp = cm.ravel()
    
    print('threshold:', th)
    print('tn, fp, fn, tp = ', tn, fp, fn, tp)
    tpr = float(tp) / (fn+tp)
    fpr = float(fp) / (tn+fp)
    print('tpr, fpr = %.4f, %.4f' % (tpr,fpr))
    
    # show
    x = np.arange(0, test_num, dtype=np.int)
    plt.figure()
    plt.plot(x, scores_test)
    plt.plot((0,test_num), (th,th))
    plt.show()
