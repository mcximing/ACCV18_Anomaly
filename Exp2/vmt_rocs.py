# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 06:34:29 2018

@author: Cong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
#plt.rcParams['figure.dpi'] = 300


if __name__ == "__main__":
    
    dm_eucl = np.load('./data/vmt_dm_eucl.npz')['arr_0']
    dm_dtw = np.load('./data/vmt_dm_dtw.npz')['arr_0']
    dm_sspd = np.load('./data/vmt_dm_sspd.npz')['arr_0']
    dm_lcss = np.load('./data/vmt_dm_lcss.npz')['arr_0']
    dm_edr = np.load('./data/vmt_dm_edr.npz')['arr_0']
    dm_erp = np.load('./data/vmt_dm_erp.npz')['arr_0']
    dm_fre = np.load('./data/vmt_dm_fre.npz')['arr_0']
    dm_hau = np.load('./data/vmt_dm_hau.npz')['arr_0']
    dm_hmm = np.load('./data/vmt_dm_hmm.npz')['arr_0']
    dm_aet = np.load('./data/vmt_dm_aet.npz')['arr_0']
    
    scores_eucl = np.sort(dm_eucl, axis=0)[1]
    scores_dtw = np.sort(dm_dtw, axis=0)[1]
    scores_sspd = np.sort(dm_sspd, axis=0)[1]
    scores_lcss = np.sort(dm_lcss, axis=0)[1]
    scores_edr = np.sort(dm_edr, axis=0)[1]
    scores_erp = np.sort(dm_erp, axis=0)[1]
    scores_fre = np.sort(dm_fre, axis=0)[1]
    scores_hau = np.sort(dm_hau, axis=0)[1]
    scores_hmm = np.sort(dm_hmm, axis=0)[1]
    scores_aet = np.sort(dm_aet, axis=0)[1]
    
    traj_num = 1500
    gt = np.zeros((traj_num,), dtype=np.int)
    gt_idx = np.loadtxt('./datasets/VMT/vmt_gt.txt', dtype=np.int)
    gt[gt_idx] = 1
    
    legends = ['Euclidean', 'DTW', 'SSPD', 'LCSS', 'EDR', 'ERP', 'Frechet', 'Hausdorff', 'HMM', 'Ours']
    scores = [scores_eucl, scores_dtw, scores_sspd, scores_lcss, scores_edr, \
              scores_erp, scores_fre, scores_hau, scores_hmm, scores_aet]
    s_num = len(scores)
    fpr = [[] for _ in range(s_num)]
    tpr = [[] for _ in range(s_num)]
    roc_auc = [0.0 for _ in range(s_num)]
    for i in range(s_num):
        fpr[i], tpr[i], _ = roc_curve(gt, scores[i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('%.4f, ' % roc_auc[i], end='')
    print('\n')
    
    plt.figure()
    for i in range(s_num):
        plt.plot(fpr[i], tpr[i], label='%s'%legends[i])#, auc=%0.6f' % (legends[i], roc_auc[i]))
    #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curves on VMT dataset')
    plt.legend(loc="lower right")
    plt.show()
