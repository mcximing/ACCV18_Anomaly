# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:43:37 2018

@author: Cong
"""


import numpy as np
import matplotlib.pyplot as plt


class Opposite():
    def __init__(self, tlen=20):
        super(Opposite, self).__init__()
        self.name = 'Opposite'
        
        x1 = np.linspace(0, 1, tlen, endpoint=False)
        x2 = x1[::-1]
        y1 = np.linspace(0, 0.6, tlen, endpoint=False)
        y2 = y1[::-1]
        self.t0 = np.c_[x1,y1]
        self.t1 = np.c_[x1,y1]
        self.t2 = np.c_[x2,y2]
        
        sigma = 1. / tlen / 5
        self.t0 += np.random.normal(0, sigma, self.t0.shape)
        self.t1 += np.random.normal(0, sigma, self.t1.shape)
        self.t2 += np.random.normal(0, sigma, self.t2.shape)


class Translation():
    def __init__(self, tlen=20, trans=0.3):
        super(Translation, self).__init__()
        self.name = 'Translation'
        
        x1 = np.linspace(0, 1, tlen, endpoint=False)
        x2 = x1
        y1 = np.linspace(0, 0.6, tlen, endpoint=False)
        y2 = y1 + trans
        self.t0 = np.c_[x1,y1]
        self.t1 = np.c_[x1,y1]
        self.t2 = np.c_[x2,y2]
        
        sigma = 1. / tlen / 5
        self.t0 += np.random.normal(0, sigma, self.t0.shape)
        self.t1 += np.random.normal(0, sigma, self.t1.shape)
        self.t2 += np.random.normal(0, sigma, self.t2.shape)


class Deviation():
    def __init__(self, tlen=20, degree=np.pi/3):
        super(Deviation, self).__init__()
        self.name = 'Deviation'
        
        x1 = np.linspace(0, 1, tlen, endpoint=False)
        x2 = x1
        y1 = np.linspace(0, 1, tlen, endpoint=False)
        
        # turn_idx + turn_num < tlen
        turn_idx = np.random.randint(8, 12) # where to turn
        turn_num = np.random.randint(4, 8) # how many points
        deg = degree / turn_num
        k = np.pi / 4
        alpha = k
        dx = 1. / tlen
        y2 = y1.copy()
        for i in range(turn_num):
            delta = alpha - deg # turn part: compute arcs
            dy = np.tan(delta) * dx
            y2[turn_idx + i] = y2[turn_idx + i-1] + dy
            alpha = delta
        for j in range(turn_idx + turn_num, tlen):
            y2[j] = y2[j-1] + np.tan(k - degree) * dx # after turn: line
        
        self.t0 = np.c_[x1,y1]
        self.t1 = np.c_[x1,y1]
        self.t2 = np.c_[x2,y2]
        
        sigma = 1. / tlen / 5
        self.t0 += np.random.normal(0, sigma, self.t0.shape)
        self.t1 += np.random.normal(0, sigma, self.t1.shape)
        self.t2 += np.random.normal(0, sigma, self.t2.shape)


class Wait():
    def __init__(self, tlen=20):
        super(Wait, self).__init__()
        self.name = 'Wait'
        
        x1 = np.linspace(0, 1, tlen, endpoint=False)
        y1 = 0.5 * np.ones_like(x1)
        
        step_gap = 1. / tlen
        wait_px, wait_span = np.random.random(2)
        wait_pts = np.ones((int(wait_span/step_gap), )) * wait_px
        x2 = np.r_[x1[x1 < wait_px], wait_pts, x1[x1 > wait_px]]
        y2 = 0.5 * np.ones_like(x2)
        
        self.t0 = np.c_[x1,y1]
        self.t1 = np.c_[x1,y1]
        self.t2 = np.c_[x2,y2]
        
        sigma = 1. / tlen / 5
        self.t0 += np.random.normal(0, sigma, self.t0.shape)
        self.t1 += np.random.normal(0, sigma, self.t1.shape)
        self.t2 += np.random.normal(0, sigma, self.t2.shape)


class Loop():
    def __init__(self, tlen=20):
        super(Loop, self).__init__()
        self.name = 'Loop'
        
        x1 = np.linspace(0, 1, tlen, endpoint=False)
        y1 = 0.5 * np.ones_like(x1)
        
        loop_idx = np.random.randint(5, 15) # where to insert
        step_gap = 1. / tlen
        radius = np.random.randint(5, 10) / 100.
        loop_num = radius * 2 * np.pi / step_gap # how many points
        degs = np.linspace(0, 2*np.pi, loop_num, endpoint=False)
        loop_pts = [radius*np.sin(degs)+(loop_idx/tlen), radius*np.cos(degs)+(0.5-radius)]
        
        x2 = np.r_[x1[:loop_idx], loop_pts[0], x1[loop_idx:]]
        y2 = np.r_[y1[:loop_idx], loop_pts[1], y1[loop_idx:]]

        self.t0 = np.c_[x1,y1]
        self.t1 = np.c_[x1,y1]
        self.t2 = np.c_[x2,y2]
        
        sigma = 1. / tlen / 5
        self.t0 += np.random.normal(0, sigma, self.t0.shape)
        self.t1 += np.random.normal(0, sigma, self.t1.shape)
        self.t2 += np.random.normal(0, sigma, self.t2.shape)


class Speed():
    def __init__(self, tlen=20, times=1.5):
        super(Speed, self).__init__()
        self.name = 'Speed'
        
        x1 = np.linspace(0, 1, tlen, endpoint=False)
        y1 = 0.5 * np.ones_like(x1)
        
        x2 = np.linspace(0, 1, tlen/times, endpoint=False)
        y2 = 0.5 * np.ones_like(x2)
        
        self.t0 = np.c_[x1,y1]
        self.t1 = np.c_[x1,y1]
        self.t2 = np.c_[x2,y2]
        
        sigma = 1. / tlen / 5
        self.t0 += np.random.normal(0, sigma, self.t0.shape)
        self.t1 += np.random.normal(0, sigma, self.t1.shape)
        self.t2 += np.random.normal(0, sigma, self.t2.shape)



if __name__ == '__main__':
    
    #classes = [Opposite, Translation, Crossing, Deviation, Wait, Loop, Speed, Part]
    classes = [Loop]
    
    for C in classes:
        pair = C(tlen=100)
        
        plt.figure()
        
        plt.subplot(211)
        plt.scatter(pair.t1[:,0], pair.t1[:,1], marker='o', c='', edgecolors='b')
        plt.axis([0., 1., 0.2, 0.8])
        plt.title(pair.name)
        
        plt.subplot(212)
        plt.scatter(pair.t2[:,0], pair.t2[:,1], marker='o', c='', edgecolors='r')
        plt.axis([0., 1., 0.2, 0.8])
        
        plt.show()
