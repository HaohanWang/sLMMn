__author__ = 'Haohan Wang'

import numpy as np
import scipy

path = '/home/haohanw/FaSTLMM_K2_Sparsity/data'

def centralize(x):
    m = np.mean(x)
    return x-m

def normalize(x):
    m = np.mean(x)
    s = np.std(x)
    return (x-m)/s

def generateData(seed, firstTime=False):
    np.random.seed(seed)

    featureNum = 50
    we = 0.1
    sigC = 1

    X = np.loadtxt(path + 'athaliana.snps.chrom1.csv', delimiter=',')
    [n, p] = X.shape

    idx = scipy.random.randint(0,p,featureNum).astype(int)
    idx = sorted(idx)
    w = 1*np.random.normal(0, 1, size=featureNum)
    ypheno = scipy.dot(X[:,idx],w)
    ypheno = ypheno.reshape(ypheno.shape[0])
    error = np.random.normal(0, 1, n)

    if firstTime:
        C = np.dot(X, X.T)
        C = centralize(C)
        Kva, Kve = np.linalg.eigh(C)
        np.savetxt('../ATData/Kva.csv', Kva, delimiter=',')
        np.savetxt('../ATData/Kve.csv', Kve, delimiter=',')
        np.savetxt('../ATData/K.csv', C, delimiter=',')
    else:
        C = np.loadtxt('../ATData/K.csv', delimiter=',')

    causal = np.array(zip(idx, w))
    np.savetxt('../ATData/causal_'+str(seed)+'.csv', causal, '%5.2f', delimiter=',')

    y = we*error + normalize(ypheno)
    np.savetxt('../ATData/K0/y_'+str(seed)+'.csv', y, '%5.2f',delimiter=',')

    yK1 = np.random.multivariate_normal(ypheno, sigC*C, size=1)
    yK1 = yK1.reshape(yK1.shape[1])
    yK1 = we*error + normalize(yK1)
    np.savetxt('../ATData/K1/y_'+str(seed)+'.csv', yK1, '%5.2f',delimiter=',')

    C2 = np.dot(C, C)
    yK2 = np.random.multivariate_normal(ypheno, sigC*C2, size=1)
    yK2 = yK2.reshape(yK2.shape[1])
    yK2 = we*error + normalize(yK2)
    np.savetxt('../ATData/K2/y_'+str(seed)+'.csv', yK2, '%5.2f',delimiter=',')

    n = np.random.randint(1, 4)
    for i in range(n):
        C = np.dot(C, C)
    yKn = np.random.multivariate_normal(ypheno, sigC*C, size=1)
    yKn = yKn.reshape(yKn.shape[1])
    yKn = we*error + normalize(yKn)
    np.savetxt('../ATData/Kn/y_'+str(seed)+'.csv', yKn, '%5.2f',delimiter=',')