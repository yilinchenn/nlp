from a3_gmm import *
from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import math
from scipy.special import logsumexp

if __name__ == "__main__":

    M = 3
    d = 2
    myTheta = theta("a", M, d)
    myTheta.omega = np.array([[1],[2], [3]], dtype='f')
    myTheta.mu = np.array([[1,2], [3,4], [6,7]], dtype='f')
    myTheta.Sigma = np.array([[5,6], [7, 8], [8,9]], dtype='f')

    X=  np.array([[2,4], [3,5]], dtype='f')

    log_Bs = np.zeros((M, X.shape[0]))
    log_Bs_ele = np.zeros((M, X.shape[0]))
    log_Bs_noPre = np.zeros((M, X.shape[0]))
    log_Ps = np.zeros((M, X.shape[0]))
    log_Ps_ele = np.zeros((M, X.shape[0]))
    preComputedForM = preComputation(myTheta, M, d)
    #print(preComputedForM)



    log_Bs = ComputeLog_Bs(M, X, myTheta)

    for x_ind, x in enumerate(X):
        for m in range(M):
            log_Bs_ele[m, x_ind] = log_b_m_x(m, x, myTheta, preComputedForM)

    for x_ind, x in enumerate(X):
        for m in range(M):
            log_Bs_noPre[m, x_ind] = log_b_m_x(m, x, myTheta, preComputedForM)


    for x_ind, x in enumerate(X):
        for m in range(M):
            #print("[%d, %d]"%(m, x_ind))
            #print(log_p_m_x(m, x, myTheta))
            log_Ps_ele[m, x_ind] = log_p_m_x(m, x, myTheta)


    print(" in test log_Bs")
    print(log_Bs_ele)
    print(log_Bs_noPre)
    print(log_Ps_ele)

    print("real")
    log_WBs = log_Bs + np.log(myTheta.omega) # M * T
    log_Ps = log_WBs - logsumexp(log_WBs, axis=0)
    print(log_Bs)
    print(log_Ps)

    # X = np.array([[1, 2], [3, 4], [5, 6]], dtype='f')
    # c = np.array([1, 2], dtype='f')
    # print(X*c)
