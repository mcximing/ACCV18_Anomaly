# -*- coding: utf-8 -*-
"""
Implementing distance definition in the paper: 
Porikli F. Trajectory distance metric using hidden markov model based representation. ECCVw, PETS 2004.
@author: Cong
"""


import numpy as np
from scipy.cluster.vq import kmeans, vq

class Hmm(object):
    __slots__ = ['N','M', 'init', 'trans', 'mix']
   
    def __init__(self):
        self.N = None
        self.M = None
        self.init = None
        self.trans = None
        self.mix = None
    
    def printself(self):
        print (self.trans)
        print (self.mix[0].weight)
        
class Mix(object):
    __slots__ = ['M', 'mean', 'var', 'weight']
    def __init__(self):
        self.M = None
        self.mean = None
        self.var = None
        self.weight = None
        
class Param(object):
    __slots__ = ['c','alpha','beta','ksai','gama']
    def __init__(self):
        self.c = None
        self.alpha = None
        self.beta = None
        self.ksai = None
        self.gama = None
        
        
def train(OneClass, M):
        
    hmm = inithmm(OneClass, M)
    
    pout = []  
    for loop in range(0, 100):
        
        hmm = baum(hmm, OneClass)
        
        # total output prob
        temp = 0
        for k in range(len(OneClass)):
            prob = viterbi(hmm, OneClass[k])[0]
            temp = temp + prob
        #print ('total output prob (log) = ', temp)
        pout.append(temp)
        
        # compare distances between two HMMs
        if loop>0:
            if abs((pout[loop]-pout[loop-1])/pout[loop]) < 5e-6:
                #print ('Converged！')
                return hmm
                
    #print ('Exit when it does not converge after 100 loops.')
    
    return hmm


def inithmm(samples, M):
    hmm = Hmm()
    N = len(M)       # state num
    hmm.N = N        # hmm state num
    hmm.M = M        # Gaussian num in each state
                                    
    # initial probability matrix
    hmm.init = np.zeros((N,1)) 
    hmm.init[0] = 1
    
    # trans prob mat
    hmm.trans = np.zeros((N,N))
    for i in range(N-1):
        hmm.trans[i][i] = 0.5
        hmm.trans[i][i+1] = 0.5
    hmm.trans[N-1,N-1] = 1
    
    # initial clustering of pdf
    # average segment
    seg_ind = []
    for k in range(len(samples)):
        T = len(samples[k]) # must be T >= 2*N
        seg_ind.append(np.r_[np.arange(0, T, np.round(T/N)), T].astype(np.int))
    
    # Kmeans cluster the vectors in each state and get a continuous mixed normal dist
    hmm.mix = []
    for i in range(N):
        # combine the vectors in same cluster and states]=
        vector = np.zeros((0, samples[0].shape[1]))
        # gather all the sample features of each state in the same model to a new feature matrix
        for k in range(len(samples)):
            seg1 = seg_ind[k][i]   # seg i start
            seg2 = seg_ind[k][i+1] # seg i+1 end
            vector = np.vstack((vector, samples[k][seg1:seg2, :]))
        mix = getmix(vector, M[i])
        hmm.mix.append(mix)
    
    return hmm


def getmix(vector, M):
    
    # K-means to M clusters
    mean = kmeans(vector, M)[0]
    nn = vq(vector, mean)[0]
    
    # only keep the items on the diagonal
    var = np.zeros((M, vector.shape[1]))
    for j in range(0,M):
        ind = (j==nn)
        tmp = vector[ind,:]
        var[j,] = np.std(tmp, axis=0, ddof=1)

    # normalize the sample numbers as weights
    weight = np.zeros((M, 1))
    for j in range(0,M):
        weight[j] = np.sum((j == nn))
    weight = weight / np.sum(weight)
    
    mix = Mix()
    mix.M = M
    mix.mean = mean
    mix.var = var ** 2
    mix.weight = weight
    return mix
    

# Baum-welch
def baum(hmm, samples):
    mix  = hmm.mix
    N = len(mix)
    K = len(samples)
    
    SIZE = samples[0].shape[1]
    
    #compute forward and backkward prob matrices. note the multi-observation seqs and underflow issues
    #print ('Computing sample parameters...')
    Allparam = []
    for k in range(1,K+1):
        #print (k,end=' ')
        param = getparam(hmm, samples[k-1])
        Allparam.append(param)
    #print ("\n")
    
    # re-estimate the trans prob matrix A: trans
    #print ('\nRe-estimating the trans prob matrix A...')
    for i in range(1,N):
        denom = 0
        for k in range(1,K+1):
            tmp = Allparam[k-1].ksai[:,i-1,:]
            denom = denom + sum(tmp.flatten())
        for j in range(i,i+2):
            nom = 0
            for k in range(1,K+1):
                tmp = Allparam[k-1].ksai[:,i-1,j-1]                
                nom = nom   + sum(tmp.flatten(1))
            hmm.trans[i-1,j-1] = nom / denom
    
    # re-estimate the parameters of Gaussian mixture
    #print ('Re-estimating Gaussian mixture model parameters...')
    for l in range(1,N+1):
        for j in range(1,hmm.M[l-1]+1):
            #print (l,j,end=' ')
            # compute the means and var of each pdf
            nommean = np.zeros((1,SIZE))
            nomvar  = np.zeros((1,SIZE))
            denom   = 0
            for k in range(1,K+1):
                T = samples[k-1].shape[0]
                for t in range(1,T+1):
                    x = samples[k-1][t-1,]
                    nommean = nommean + Allparam[k-1].gama[t-1,l-1,j-1] * x
                    nomvar  = nomvar  + Allparam[k-1].gama[t-1,l-1,j-1] * (x-mix[l-1].mean[j-1,])**2
                    denom   = denom   + Allparam[k-1].gama[t-1,l-1,j-1]
            hmm.mix[l-1].mean[j-1,] = nommean / denom
            hmm.mix[l-1].var[j-1,] = nomvar  / denom
            
            # compute the weights of each pdf
            nom   = 0
            denom = 0
            for k in range(1,K+1):
                tmp = Allparam[k-1].gama[:,l-1,j-1]
                nom   = nom   + sum(tmp.flatten(1))
                tmp = Allparam[k-1].gama[:,l-1,]
                denom = denom + sum(tmp.flatten(1))
            hmm.mix[l-1].weight[j-1] = nom/denom
        #print ('\n')
    return hmm
        
       
def getparam(hmm, O):
    # given output seq O, compute forward prob alpha, backward prob beta, parameter c, ksai, and gama
    # Inputs:
    # hmm -- HMM model params
    # O   -- n*d observation seq
    # Output:
    # param -- params
    T = O.shape[0]   # seq length (rows), actually frame number
   
    init  = hmm.init    # initial prob
    trans = hmm.trans	# trans prob
    mix   = hmm.mix	    # Gaussian mixture
    N     = hmm.N 	    # HMM state num
    
    # given observation seq O, compute the forward prb alpha
    alpha = np.zeros((T,N))   # T is frame number
    # first frame
    x = O[0,]
    for i in range(1,N+1):
        alpha[0,i-1] = init[i-1].dot(mixture(mix[i-1],x))
    # forward prob at t = 1
    c = np.zeros((T,1))
    c[0] = 1/sum(alpha[0,])
    alpha[0,] = c[0]*(alpha[0,])
    #print alpha[0]
    #forward prob at t=2:T
    for t in range(2,T+1):
        for i in range(1,N+1):
            temp = 0
            for j in range(1,N+1):
                temp = temp + alpha[t-2,j-1] * trans[j-1,i-1]
            alpha[t-1,i-1] = temp * mixture(mix[i-1],O[t-1,])
        c[t-1]= 1/sum(alpha[t-1,])
        alpha[t-1,] = c[t-1] * alpha[t-1,]
    
    # given observation seqO, compute the backward prb beta
    beta = np.zeros((T,N))    
    # backward prob at t=T
    for l in range(1,N+1):
        beta[T-1,l-1] = c[T-1]    
    # backward probs at t=T-1:1
    for t in range(T-1,0,-1):
        x = O[t,]
        for i in range(1,N+1):
            for j in range(1,N+1):
                beta[t-1,i-1] = beta[t-1,i-1] + (beta[t,j-1]) * mixture(mix[j-1],x) * (trans[i-1,j-1])
        
        if c[t-1] < 1.0:
            beta[t-1,] = c[t-1] * beta[t-1,] # original computation
        else:
            overjudge = np.finfo(np.float64).max / c[t-1] # in case of overflow
            if overjudge < np.max(beta[t-1,]):
                #print (c[t-1])
                overidx = beta[t-1,] > (overjudge)
                beta[t-1,][overidx] = np.finfo(np.float64).max
                beta[t-1,][np.bitwise_not(overidx)] = c[t-1] * beta[t-1,][np.bitwise_not(overidx)]
            else:
                beta[t-1,] = c[t-1] * beta[t-1,] # original computation
    
    # ksai
    ksai = np.zeros((T-1,N,N))
    for t in range(1,T):
        denom = sum(alpha[t-1,]*beta[t-1,])
        for i in range(1,N):
            for j in range(i ,i+2):
                nom = alpha[t-1,i-1] * trans[i-1,j-1] * mixture(mix[j-1],O[t,]) * beta[t,j-1]
                ksai[t-1,i-1,j-1] = c[t-1] * nom/denom
                
    # mixture output prob: gama
    gama = np.zeros((T,N,max(hmm.M)))
    for t in range(1,T+1):
        pab = np.zeros((N,1))
        for l in range(1,N+1):
            pab[l-1] = alpha[t-1,l-1] * beta[t-1,l-1]
        x = O[t-1,]
        for l in range(1,N+1):
            prob = np.zeros((mix[l-1].M,1))
            for j in range(1,mix[l-1].M+1):
                m = mix[l-1].mean[j-1,]
                v = mix[l-1].var[j-1,]
                prob[j-1] = mix[l-1].weight[j-1] * pdf(m, v, x)
            tmp  = pab[l-1]/sum(pab)
            for j in range(1,mix[l-1].M+1):
                gama[t-1,l-1,j-1] = tmp.dot(prob[j-1])/sum(prob)
                
    param = Param()
    param.c     = c
    param.alpha = alpha
    param.beta  = beta
    param.ksai  = ksai
    param.gama  = gama
    
    return param
    

def mixture(mix, x):
    # compute the output prob of an obserrvation vector x at an HMM state
    # compute the output prob
    # Input:
    #   mix  -- Gaussian mixture
    #   x    -- input vector, SIZE*1
    # Output:
    #   prob -- output prob
    M = mix.M
    prob = 0.0
    for j in range(1,M + 1):
        m = mix.mean[j-1,]
        v = mix.var[j-1,]
        w = mix.weight[j-1]
        prob = prob + w * pdf(m, v, x)
    
    return prob
        

def pdf(m, v, x):
    # single Gaussian prob density function (p.d.f.)
    #Input:
    #m -- mean vector, SIZE*1
    #v -- var vector, SIZE*1
    #x -- input vector, SIZE*1
    #Output:
    #p -- output prob
    eps = np.finfo(np.float64).tiny
    if v.min() < eps or np.prod(v) < eps:
        return 1.0 # in case of ZeroDivisionError
    p = (2 * np.pi * np.prod(v)) ** -0.5 * np.exp(-0.5 * ((x-m)/v).dot((x-m).T))
    if p < eps:
        p = eps # in case of overflow
    return p 
    

def viterbi(hmm, O):
    # Viterbi algorithm
    # given output seq O, compute the forward prob delta and backward prob fai
    #Input:
    #  hmm -- hmm model
    #  O   -- input observation seq, N*D, D is the dim of vector
    #Output:
    #  prob -- output prob
    #  q    -- state seq
    init  = hmm.init.copy()
    trans = hmm.trans.copy()
    mix   = hmm.mix	     # Gaussian mixture
    N     = hmm.N	     # HMM state num
    T     = O.shape[0]
    
    # compute log(init)
    ind1  = (init>0).nonzero()[0]
    ind0  = (init<=0).nonzero()[0]
    init[ind0] = np.NINF
    init[ind1] = np.log(init[ind1])
    
    # compute log(trans)
    ind1 = (trans>0).nonzero()
    ind0 = (trans<=0).nonzero()
    trans[ind0] = np.NINF
    trans[ind1] = np.log(trans[ind1])
    
    # initialization
    delta = np.zeros((T,N))
    fai   = np.zeros((T,N))
    q     = np.zeros((T,1))
    
    #t=1  viterbi initialization 当t=1
    x = O[0,:]
    for i in range(1,N+1):
        delta[0,i-1] = init[i-1]+ np.log(mixture(mix[i-1],x))
    
    #t=2:T viterbi iterationo t>2
    for t in range(2,T+1):
        for j in range(1,N+1):
            delta[t-1,j-1] = np.max(delta[t-2,] + trans[:,j-1].T)
            fai[t-1,j-1] = np.argmax(delta[t-2,] + trans[:,j-1].T)
            x = O[t-1,:]
            delta[t-1,j-1] = delta[t-1,j-1] + np.log(mixture(mix[j-1],x))
            
    # final prob
    prob = np.max(delta[T-1,:])
    q[T-1] = np.argmax(delta[T-1,:])

    # best state path
    for t in range(T-1,2,-1):
        q[t-1] = fai[t, int(q[t]-1)]
    
    return prob, q


def hmmdist(sample0, sample1, M=np.array([1,1,1,1,1,1])):
    hmm0 = train([sample0], M)
    hmm1 = train([sample1], M)
    prob00 = viterbi(hmm0, sample0)[0]
    prob01 = viterbi(hmm0, sample1)[0]
    prob10 = viterbi(hmm1, sample0)[0]
    prob11 = viterbi(hmm1, sample1)[0]
    #print(prob00, prob01, prob10, prob11)
    dist = prob01 + prob10 - prob00 - prob11
    return np.abs(dist)


def hmmdist_eval(hmm0, hmm1, sample0, sample1):
    prob00 = viterbi(hmm0, sample0)[0]
    prob01 = viterbi(hmm0, sample1)[0]
    prob10 = viterbi(hmm1, sample0)[0]
    prob11 = viterbi(hmm1, sample1)[0]
    #print(prob00, prob01, prob10, prob11)
    dist = prob01 + prob10 - prob00 - prob11
    return np.abs(dist)


if __name__ == "__main__":
    
    M = np.array([1,1,1])  # hmm: 3 states, 1 GMM
    
    print ('Loading data...')
    from datasets import read_cross_dataset#, read_traffic_dataset
    train_set, train_labels, test_set, true_labels = read_cross_dataset()
    sample0 = train_set[20]
    sample1 = train_set[25]
    sample2 = train_set[100]
    
    print ('Training...')
    print(hmmdist(sample0, sample1))
    print(hmmdist(sample1, sample2))
    print(hmmdist(sample1, sample1))
    
#    #save hmm models
#    f = open('myhmm.dat','wb')
#    pickle.dump(hmm1, f, -1)
#    f.close()
    
    print ('Completed')
