__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
import scipy

# path = '/home/haohanw/FaSTLMM_K2_Sparsity/data/'
path = '../ATData/'

from utility.simpleFunctions import *

def generateData(seed, firstTime=False, test=False):
    plt = None
    if test:
        from matplotlib import pyplot as plt
    np.random.seed(seed)

    featureNum = 100
    we = 0
    sigC = 1

    X = np.loadtxt(path + 'athaliana2.snps.chrom5.csv', delimiter=',')
    [n, p] = X.shape
    X = reOrder(X)

    idx = scipy.random.randint(0, p, featureNum).astype(int)
    idx = sorted(idx)
    w = 1 * np.random.normal(0, 1, size=featureNum)
    ypheno = scipy.dot(X[:, idx], w)
    ypheno = (ypheno - ypheno.mean()) / ypheno.std()
    ypheno = ypheno.reshape(ypheno.shape[0])
    error = np.random.normal(0, 1, n)

    if firstTime:
        C = np.dot(X, X.T)
        Kva, Kve = np.linalg.eigh(C)
        np.savetxt('../ATData/Kva.csv', Kva, delimiter=',')
        np.savetxt('../ATData/Kve.csv', Kve, delimiter=',')
        np.savetxt('../ATData/K.csv', C, delimiter=',')
    else:
        C = np.loadtxt('../ATData/K.csv', delimiter=',')
        Kva = np.loadtxt('../ATData/Kva.csv', delimiter=',')

    C1 = rescale(C)
    C1 = C
    # if test:
    #     plt.imshow(C1)
    #     print C1
    #     plt.show()

    if test:
        ind = np.array(xrange(Kva.shape[0]))
        plt.scatter(ind[:-1], mapping2ZeroOne(Kva[:-1]), color='y', marker='+')
        plt.scatter(ind[:-1], mapping2ZeroOne(np.power(Kva, 2)[:-1]), color='b', marker='+')
        plt.scatter(ind[:-1], mapping2ZeroOne(np.power(Kva, 4)[:-1]), color='m', marker='+')
        plt.show()

    causal = np.array(zip(idx, w))
    if not test:
        np.savetxt('../ATData/causal_'+str(seed)+'.csv', causal, '%5.2f', delimiter=',')

    y = we * error + normalize(ypheno)
    if not test:
        np.savetxt('../ATData/K0/y_'+str(seed)+'.csv', y, '%5.2f', delimiter=',')

    yK1 = np.random.multivariate_normal(ypheno, sigC * C1, size=1)
    yK1 = yK1.reshape(yK1.shape[1])
    yK1 = we * error + normalize(yK1)
    if not test:
        np.savetxt('../ATData/K1/y_'+str(seed)+'.csv', yK1, '%5.2f', delimiter=',')

    C2 = np.dot(C, C)
    # C2 = rescale(C2)
    yK2 = np.random.multivariate_normal(ypheno, sigC * C2, size=1)
    yK2 = yK2.reshape(yK2.shape[1])
    yK2 = we * error + normalize(yK2)
    # if test:
    #     plt.imshow(C2)
    #     plt.show()
    #     print C2
    if not test:
        np.savetxt('../ATData/K2/y_'+str(seed)+'.csv', yK2, '%5.2f', delimiter=',')

    C3 = np.dot(C2,C)
    # C3 = rescale(C3)
    # if test:
    #     plt.imshow(C3)
    #     plt.show()
    #     print C3
    yKn = np.random.multivariate_normal(ypheno, sigC * C3, size=1)
    yKn = yKn.reshape(yKn.shape[1])
    yKn = we * error + normalize(yKn)
    if not test:
        np.savetxt('../ATData/Kn/y_'+str(seed)+'.csv', yKn, '%5.2f', delimiter=',')

    if test:
        x = xrange(len(y))
        plt.scatter(x, y, color='g')
        plt.scatter(x, yK1, color='y')
        plt.scatter(x, yK2, color='b')
        plt.scatter(x, yKn, color='m')
        plt.show()

        # def imshowY(y):
        #     y = y.reshape(y.shape[0], 1)
        #     plt.imshow(np.dot(y, y.T))
        #     plt.show()
        #
        # imshowY(y)
        # imshowY(yK1)
        # imshowY(yK2)
        # imshowY(yKn)

if __name__ == '__main__':
    generateData(0, False, True)