# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:10:32 2018

@author: Cong
"""

import numpy as np
import scipy.io as sio
from scipy import interpolate


##===TRAFFIC===##

def read_traffic_dataset(filename='D:/Data/Anomaly/TRAFFIC/TRAFFIC.mat', \
                         randomize=False, read_name='traffic_set_idx.txt', \
                         normalize=True, resample=False, tlen=12):
    matdata = sio.loadmat(filename)
    
    tracks_traffic_data = matdata['tracks_traffic']
    truth = matdata['truth'] # (300, 2)
    
    tracks_traffic = []
    for data in tracks_traffic_data:
        tracks_traffic.append(data[0])
    tracks_traffic = np.array(tracks_traffic).transpose(0,2,1) # (300, 50, 2)
    
    if resample:
        # spline fitting
        t = np.linspace(0, 1, num=tlen) # fixed-length 
        tracks_traffic_data = []
        for traj in tracks_traffic:
            tck,u = interpolate.splprep(traj.T, k=3, s=0) # traj:(M,2)
            out = interpolate.splev(t,tck) # a list saves two arrays
            tracks_traffic_data.append(np.c_[out[0], out[1]]) # (T,2)
        tracks_traffic = np.array(tracks_traffic_data) # (300, 50, 2)
    
    if normalize:
        temp = tracks_traffic.T # (2, 50, 300)
        xmin = np.min(temp[0])
        xmax = np.max(temp[0])
        ymin = np.min(temp[1])
        ymax = np.max(temp[1])
        x = (temp[0] - xmin) / (xmax - xmin) # [0,1]
        y = (temp[1] - ymin) / (ymax - ymin)
        tracks_traffic = np.array([x,y]).T # (300, 50, 2)
    
    #train_num = 150
    #test_num = 150
    if randomize:
        idx = np.arange(0, 200, dtype=np.uint)
        np.random.shuffle(idx)
        import datetime
        s_time = datetime.datetime.strftime(datetime.datetime.now(),'%m%d%H%M')
        np.savetxt('traffic_set_idx_%s.txt' % s_time, idx, fmt='%d')
    else:
        idx = np.loadtxt(read_name, dtype=np.uint)
    train_idx = idx[:150]
    test_idx = np.r_[idx[-50:], np.arange(200, 300, dtype=np.uint)]
    
    train_samples = tracks_traffic[train_idx]
    train_labels = truth[train_idx, 0]
    
    test_samples = tracks_traffic[test_idx]
    truth_labels = truth[test_idx, 0]
    truth_labels[truth_labels>7] = 8
    
    return train_samples, train_labels, test_samples, truth_labels


##===CROSS===##

def load_cvrr_train(datadir):
    ## load training trajectories
    filename = datadir +  'train.mat'
    matdata = sio.loadmat(filename)
    tracks_train_cells = matdata['tracks_train'] #1900*1 cell, each cell 2*N double
    labels_train = matdata['labels_train'].ravel().T # 1*1900 double
    #ind_tracks_filt_l = matdata['ind_tracks_filt_l'].ravel() # 1900*1 logical
    #ind_tracks_clust_l = matdata['ind_tracks_clust_l'].ravel()# 1900*1 logical
    ind_tracks_model_l = matdata['ind_tracks_model_l'].ravel() # 1900*1 logical
    
    # parse MATLAB cell data
    datalen = len(tracks_train_cells)
    #traj_shape = tracks_train_cells[0][0].shape
    tracks_train = [] # 1900*1 list, each item 2*N array
    for i in range(datalen):
        tracks_train.append(tracks_train_cells[i][0]) # unfold the two nests

    return tracks_train, labels_train, ind_tracks_model_l


def load_cvrr_test(datadir):
    ## load test trajectories
    filename = datadir +  'test.mat'
    matdata = sio.loadmat(filename)
    
    tracks_cells = matdata['tracks'] # 9700*1 cell, each cell 2*N double array
    labels = matdata['labels'].T # 3*9700
    abnormal_offline = matdata['abnormal_offline'].ravel() # 9700*1 logical array
    abnormal_online_cells = matdata['abnormal_online'] # 9700*1 cell, each cell N*1 double array
    #step = int(matdata['step'])
    
    # parse MATLAB cell data
    datalen_test = len(tracks_cells)
    tracks_test = [] # 9700*1 list, each item 2*N array
    for i in range(datalen_test):
        tracks_test.append(tracks_cells[i][0])
    abnormal_online = []
    for i in range(datalen_test):
        abnormal_online.append(abnormal_online_cells[i][0].ravel())

    return (tracks_test, labels, abnormal_offline)


def read_cross_dataset(datadir='D:/Data/Anomaly/CVRR/CVRR_dataset_trajectory_analysis_v0/CROSS/', \
                       resample=True, normalize=True, tlen=12):

    tracks_train, labels_train, filters_train = load_cvrr_train(datadir)
    tracks_test, labels_test, abnormal_offline = load_cvrr_test(datadir)
    
    train_labels = labels_train
    truth_labels = labels_test[:, 2]
    
    if resample:
        # spline fitting
        t = np.linspace(0, 1, num=tlen) # fixed-length 
        trainset = []
        for traj in tracks_train:
            tck,u = interpolate.splprep(traj, k=3, s=0) # traj:(2,M)
            out = interpolate.splev(t,tck) # a list saves two arrays
            trainset.append(np.c_[out[0], out[1]]) # (T,2)
        testset = []
        for traj in tracks_test:
            tck,u = interpolate.splprep(traj, k=3, s=0) # traj:(2,M)
            out = interpolate.splev(t,tck) # a list saves two arrays
            testset.append(np.c_[out[0], out[1]]) # (T,2)
        
        del tracks_train
        del tracks_test
        train_samples = np.array(trainset) # (1900,T,2)
        test_samples = np.array(testset) # (9700,T,2)
        
        if normalize:
            xmax = max(np.amax(train_samples.T[0]), np.amax(test_samples.T[0]))
            ymax = max(np.amax(train_samples.T[1]), np.amax(test_samples.T[1]))
            xmin = min(np.amin(train_samples.T[0]), np.amin(test_samples.T[0]))
            ymin = min(np.amin(train_samples.T[1]), np.amin(test_samples.T[1]))
            
            train_samples.T[0] -= xmin
            train_samples.T[0] /= (xmax - xmin)
            train_samples.T[1] -= ymin
            train_samples.T[1] /= (ymax - ymin)
            
            test_samples.T[0] -= xmin
            test_samples.T[0] /= (xmax - xmin)
            test_samples.T[1] -= ymin
            test_samples.T[1] /= (ymax - ymin)
            
    else:
        if normalize:
            # get the min-max value of x and y coordinates
            xmax = ymax = 0
            xmin = ymin = 2147483647
            for traj in tracks_train:
                x1, y1 = np.amax(traj, axis=1) # (2,N)
                x0, y0 = np.amin(traj, axis=1)
                if x1 > xmax: xmax = x1
                if y1 > ymax: ymax = y1
                if x0 < xmin: xmin = x0
                if y0 < ymin: ymin = y0
            for traj in tracks_test:
                x1, y1 = np.amax(traj, axis=1)
                x0, y0 = np.amin(traj, axis=1)
                if x1 > xmax: xmax = x1
                if y1 > ymax: ymax = y1
                if x0 < xmin: xmin = x0
                if y0 < ymin: ymin = y0
            
            # normalization
            xscale = xmax - xmin
            yscale = ymax - ymin
            
            train_samples = []
            for traj in tracks_train:
                x,y = traj
                traj[0,:] = (x - xmin) / xscale
                traj[1,:] = (y - ymin) / yscale
                train_samples.append(traj.T)
            test_samples = []
            for traj in tracks_test:
                x,y = traj
                traj[0,:] = (x - xmin) / xscale
                traj[1,:] = (y - ymin) / yscale
                test_samples.append(traj.T)
        else:
            train_samples = []
            for traj in tracks_train:
                train_samples.append(traj.T)
            test_samples = []
            for traj in tracks_test:
                test_samples.append(traj.T)
        del tracks_train
        del tracks_test
    
    return train_samples, train_labels, test_samples, truth_labels


##===trajectory datasets===##
# all the data are in [(N,2)] shape.
    
def read_VMT(filename='D:/Data/trj/CASIA_tjc.mat', normalize=True, resample=True, tlen=12):
    matdata = sio.loadmat(filename)
    labels = matdata['labels'].ravel()
    traj_cells = matdata['tjc']
    trajs = []
    for cell in traj_cells:
        trajs.append(cell[0]) # unfold the two nests
    
    if resample:
        t = np.linspace(0, 1, num=tlen) # fixed-length
        dataset = []
        for traj in trajs:
            traj = traj.T
            tck,u = interpolate.splprep(traj, k=3, s=0) # traj:(2,M)
            out = interpolate.splev(t,tck) # a list saves two arrays
            dataset.append(np.c_[out[0], out[1]]) # (T,2)
        trajs = np.array(dataset) # (1500,T,2)
    
    if normalize:
        # normalization to 0-1
        for traj in trajs:
            x,y = traj.T
            traj[:,0] = x / 320.
            traj[:,1] = y / 240.
    
    return (trajs, labels)
    


if __name__ == '__main__':
    train_set, train_labels, test_set, true_labels = read_cross_dataset(resample=False)
    sample0 = train_set[0]
    print(sample0.shape)

