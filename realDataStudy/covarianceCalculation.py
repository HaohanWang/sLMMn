__author__ = 'Haohan Wang'

import numpy as np

def calculateCovariance():
    Ks = []
    K = np.loadtxt('../ATData/K.csv', delimiter=',')
    Ks.append(K)
    K2 = np.dot(K,K)
    Ks.append(K2)
    Ks.append(np.dot(K2, K))
    np.save('../ATData/Ks', Ks)


if __name__ == '__main__':
    calculateCovariance()