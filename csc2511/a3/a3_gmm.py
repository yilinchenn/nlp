from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.full((M, 1), 1.0 / M)
        self.mu = np.zeros((M, d))
        self.Sigma = np.full((M, d), 1.0)


def preComputation(myTheta, M, d):
    preComputation = []
    for m in range(M):
        term1 = np.sum(np.divide(np.square(myTheta.mu[m]), 2.0 * myTheta.Sigma[m]))
        term3 = np.prod(myTheta.Sigma[m])
        result = term1 + np.log(2.0 * np.pi) * d / 2.0 + 1 / 2.0 * np.log(term3)
        preComputation.append(result)

    return preComputation


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    d = myTheta.Sigma.shape[1]

    c = (myTheta.mu[m] * 1. / myTheta.Sigma[m])
    b = np.sum(x * c)
    result1 = 1. / 2 * np.sum(np.square(x) * 1. / myTheta.Sigma[m]) - b

    if len(preComputedForM) != 0:
        final_result = - result1 - preComputedForM[m]
    else:
        # calculate the 2nd term
        term1 = np.sum(np.divide(np.square(myTheta.mu[m]), 2.0 * myTheta.Sigma[m]))
        term3 = np.prod(myTheta.Sigma[m])
        result2 = term1 + np.log(2.0 * np.pi) * d / 2.0 + 1 / 2.0 * np.log(term3)
        final_result = - result1 - result2

    return final_result


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    preComputedForM = preComputation(myTheta, myTheta.Sigma.shape[0], myTheta.Sigma.shape[1])

    log_Bs = []
    for k in range(myTheta.Sigma.shape[0]):  # M
        log_Bs.append(log_b_m_x(k, x, myTheta, preComputedForM))

    log_Bs = np.array(log_Bs)

    x = logsumexp(log_Bs + np.log(myTheta.omega.transpose()))
    log_p = np.log(myTheta.omega[m]) + log_Bs[m] - x

    return log_p


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''

    log_Ps = logsumexp(np.log(myTheta.omega) + log_Bs, axis=0)
    result = np.sum(log_Ps)

    return result


def ComputeLog_Bs(M, X, myTheta):
    T = X.shape[0]
    log_Bs = np.zeros((M, T))
    preComputedForM = preComputation(myTheta, M, X.shape[1])

    for m in range(M):
        # row = log_b_m_x(m, X, myTheta, preComputedForM)
        c = (myTheta.mu[m] * 1. / myTheta.Sigma[m])
        b = np.sum(X * c, axis=1)
        result1 = 1. / 2 * np.sum(np.square(X) * 1. / myTheta.Sigma[m], axis=1) - b
        final_result = - result1 - preComputedForM[m]
        log_Bs[m] = final_result

    return log_Bs


def ComputeLog_Ps(log_Bs, myTheta):
    log_WBs = log_Bs + np.log(myTheta.omega)  # M * T
    log_Ps = log_WBs - logsumexp(log_WBs, axis=0)
    return log_Ps


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    # # initialize theta
    myTheta = theta(speaker, M, X.shape[1])
    myTheta.mu = X[np.random.choice(X.shape[0], M, replace=False)]
    # myTheta.mu = X[0:M]

    i = 0
    pre_L = np.NINF
    improvement = np.Infinity
    T = X.shape[0]

    while (i < maxIter and improvement > epsilon):
        print("**********************iteration******************************" + str(i))
        # print(X.shape)
        log_Bs = ComputeLog_Bs(M, X, myTheta)  # M * T
        log_Ps = ComputeLog_Ps(log_Bs, myTheta)  # M * T
        L = logLik(log_Bs, myTheta)

        # print(log_Bs)
        # print(log_Ps)
        Ps = np.exp(log_Ps)

        # update theta
        for m in range(M):
            myTheta.omega[m] = np.sum(Ps[m]) / T
            myTheta.mu[m] = np.dot(Ps[m], X) / np.sum(Ps[m])
            myTheta.Sigma[m] = np.subtract(np.divide(np.dot(Ps[m], np.square(X)), np.sum(Ps[m])),
                                           np.square(myTheta.mu[m]))

        improvement = L - pre_L
        pre_L = L
        i += 1

        # print(L)
        # print(improvement)
        # print("T is ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ %d"%T)
        # #print(pre_L)
        # print("theta")
        # print(myTheta.omega)
        # print(myTheta.mu)
        # print(myTheta.Sigma)

    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    M = models[0].mu.shape[0]
    logs = []
    for i in range(len(models)):
        log_Bs = ComputeLog_Bs(M, mfcc, models[i])
        x = logLik(log_Bs, models[i])
        logs.append(x)

    out = open("gmmLiks.txt", "a")
    # out = open("nothing.txt", "w")

    logs_m = np.array(logs)
    idx = (-logs_m).argsort()

    if k > 0:
        out.write(models[correctID].name)
        out.write("\n")
        for i in range(k):
            out.write(models[idx[i]].name + " " + str(logs[idx[i]]))
            out.write("\n")

    out.close()

    bestModel = idx[0]

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    count = 0
    # d = 13
    # k = 5  # number of top speakers to display, <= 0 if none
    # M = 8
    # epsilon = 0.0
    # maxIter = 1

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # if count >= 32:
            #     break
            print(speaker)
            count += 1

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))
            # break

    out = open("gmmLiks.txt", "w")
    out.close()

    # evaluate 
    numCorrect = 0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)

    print("accuracy %f \n" % accuracy)

