# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 08:39:23 2018

@author: Cong
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
    
    namelist = []
    with open('D:/Data/DETRAC/train-names.txt', 'r') as fp:
        for line in fp:
            namelist.append(line.strip())
    
    with open('D:/Data/DETRAC/train-traj-data/label.pkl', 'rb') as fp:
        labels = pickle.load(fp)
    
    # 31 sets with positive samples
    idx = []
    for i in range(60):
        name = namelist[i]
        if (labels[name]==1).sum() > 0:
            idx.append(i)
    print(len(idx))
    
    with open('./data/detrac_dms_eucl.pkl', 'rb') as fp:
        dms_eucl = pickle.load(fp)
    with open('./data/detrac_dms_dtw.pkl', 'rb') as fp:
        dms_dtw = pickle.load(fp)
    with open('./data/detrac_dms_sspd.pkl', 'rb') as fp:
        dms_sspd = pickle.load(fp)
    with open('./data/detrac_dms_lcss.pkl', 'rb') as fp:
        dms_lcss = pickle.load(fp)
    with open('./data/detrac_dms_edr.pkl', 'rb') as fp:
        dms_edr = pickle.load(fp)
    with open('./data/detrac_dms_erp.pkl', 'rb') as fp:
        dms_erp = pickle.load(fp)
    with open('./data/detrac_dms_fre.pkl', 'rb') as fp:
        dms_fre = pickle.load(fp)
    with open('./data/detrac_dms_hau.pkl', 'rb') as fp:
        dms_hau = pickle.load(fp)
    with open('./data/detrac_dms_hmm.pkl', 'rb') as fp:
        dms_hmm = pickle.load(fp)
    with open('./data/detrac_dms_aet.pkl', 'rb') as fp:
        dms_aet = pickle.load(fp)

    scores_eucl = []
    scores_dtw = []
    scores_sspd = []
    scores_lcss = []
    scores_edr = []
    scores_erp = []
    scores_fre = []
    scores_hau = []
    scores_hmm = []
    scores_aet = []

    for dm in dms_eucl:
        scores_eucl.append(np.sort(dm, axis=0)[1])
    for dm in dms_dtw:
        scores_dtw.append(np.sort(dm, axis=0)[1])
    for dm in dms_sspd:
        scores_sspd.append(np.sort(dm, axis=0)[1])
    for dm in dms_lcss:
        scores_lcss.append(np.sort(dm, axis=0)[1])
    for dm in dms_edr:
        scores_edr.append(np.sort(dm, axis=0)[1])
    for dm in dms_erp:
        scores_erp.append(np.sort(dm, axis=0)[1])
    for dm in dms_fre:
        scores_fre.append(np.sort(dm, axis=0)[1])
    for dm in dms_hau:
        scores_hau.append(np.sort(dm, axis=0)[1])
    for dm in dms_hmm:
        scores_hmm.append(np.sort(dm, axis=0)[1])
    for dm in dms_aes:
        scores_aes.append(np.sort(dm, axis=0)[1])
    for dm in dms_aes2:
        scores_aes2.append(np.sort(dm, axis=0)[1])
    for dm in dms_aet:
        scores_aet.append(np.sort(dm, axis=0)[1])
    
    # compute average auc
    auc_data = []
    for s in range(31):
        name = namelist[idx[s]]
        scores = [scores_eucl[s], scores_dtw[s], scores_sspd[s], scores_lcss[s], scores_edr[s], \
                  scores_erp[s], scores_fre[s], scores_hau[s], scores_hmm[s], scores_aet[s]]
        gt = labels[name]
        mask = gt<2
        gt = gt[mask]
        s_num = len(scores)
        fpr = [[] for _ in range(s_num)]
        tpr = [[] for _ in range(s_num)]
        roc_auc = np.zeros((s_num,))
        for i in range(s_num):
            scores[i] = scores[i][mask]
            fpr[i], tpr[i], _ = roc_curve(gt, scores[i], drop_intermediate=False)
            roc_auc[i] += auc(fpr[i], tpr[i])
        auc_data.append(roc_auc)
    auc_data = np.array(auc_data)
    aucs = auc_data.mean(axis=0)
    for i in range(s_num):
        print('%.4f, ' % aucs[i], end='')
    