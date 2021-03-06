# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:13:59 2018

@author: Cong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
    res_dir = './data/traffic_set_idx_06210519/'
    
    dm_eucl = np.load(res_dir+'traffic_dm_eucl.npz')['arr_0']
    dm_dtw = np.load(res_dir+'traffic_dm_dtw.npz')['arr_0']
    dm_sspd = np.load(res_dir+'traffic_dm_sspd.npz')['arr_0']
    dm_lcss = np.load(res_dir+'traffic_dm_lcss.npz')['arr_0']
    dm_edr = np.load(res_dir+'traffic_dm_edr.npz')['arr_0']
    dm_erp = np.load(res_dir+'traffic_dm_erp.npz')['arr_0']
    dm_fre = np.load(res_dir+'traffic_dm_fre.npz')['arr_0']
    dm_hau = np.load(res_dir+'traffic_dm_hau.npz')['arr_0']
    dm_hmm = np.load(res_dir+'traffic_dm_hmm.npz')['dist_mat']
    dm_aet = np.load(res_dir+'traffic_dm_aes.npz')['dist_mat']
    
    scores_eucl = np.amin(dm_eucl, axis=0)
    scores_dtw = np.amin(dm_dtw, axis=0)
    scores_sspd = np.amin(dm_sspd, axis=0)
    scores_lcss = np.amin(dm_lcss, axis=0)
    scores_edr = np.amin(dm_edr, axis=0)
    scores_erp = np.amin(dm_erp, axis=0)
    scores_fre = np.amin(dm_fre, axis=0)
    scores_hau = np.amin(dm_hau, axis=0)
    scores_hmm = np.amin(dm_hmm, axis=0)
    scores_aet = np.amin(dm_aet, axis=0)
    
    test_num = 150
    gt_idx = np.arange(50, test_num, dtype=np.uint)
    gt = np.zeros((test_num,))
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
    
    plt.figure()
    for i in range(s_num):
        plt.plot(fpr[i], tpr[i], label='%s, auc=%0.6f' % (legends[i], roc_auc[i]))
    #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves on TRAFFIC dataset')
    plt.legend(loc="lower right")
    plt.show()

