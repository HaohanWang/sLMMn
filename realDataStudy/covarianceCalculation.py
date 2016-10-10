__author__ = 'Haohan Wang'

import sys

sys.path.append('../')

from utility.simpleFunctions import *

def splitIntoChroms():
    path = '/home/haohanw/FaSTLMM_K2_Sparsity/data/'
    pos = np.loadtxt(path+'athaliana.snps.chromPositionInfo.txt', delimiter='\t')
    X = np.loadtxt(path + 'athaliana.snps.all.csv', delimiter=',')
    for i in range(1, 6):
        snps = X[:, pos==i]
        np.savetxt(path+'athaliana.snps.chrom'+str(i)+'.csv', snps, delimiter=',', fmt='%d')

def calculateCovariance_AT():
    path = '/home/haohanw/FaSTLMM_K2_Sparsity/data/'
    pos = np.loadtxt(path+'athaliana.snps.chromPositionInfo.txt', delimiter='\t')
    X = np.loadtxt(path + 'athaliana.snps.all.csv', delimiter=',')
    X = X[:, pos==2]
    K = np.dot(X, X.T)
    Kva, Kve = np.linalg.eigh(K)
    np.savetxt('../ATData/Kva.csv', Kva, delimiter=',')
    np.savetxt('../ATData/Kve.csv', Kve, delimiter=',')
    np.savetxt('../ATData/K.csv', K, delimiter=',')
    Ks = []
    Ks.append(K)
    K2 = np.dot(K, K)
    Ks.append(K2)
    Ks.append(np.dot(K2, K))
    np.save('../ATData/Ks', Ks)

def phenoCovariance_AT():
    path = '/home/haohanw/FaSTLMM_K2_Sparsity/data/'
    y = np.nan_to_num(np.loadtxt(path + 'athaliana2.phenos.csv', delimiter=','))
    K = np.dot(y, y.T)
    Ks = []
    Ks.append(K)
    K2 = np.dot(K, K)
    Ks.append(K2)
    Ks.append(np.dot(K2, K))
    np.save('../ATData/yKs', Ks)

def calculateCovariance_Alz():
    path = '/home/haohanw/AlzheimerPreprocess/generateData/'
    X = np.loadtxt(path + 'snps.csv', delimiter=',')
    K = np.dot(X, X.T)
    Kva, Kve = np.linalg.eigh(K)
    np.savetxt('../AlzData/Kva.csv', Kva, delimiter=',')
    np.savetxt('../AlzData/Kve.csv', Kve, delimiter=',')
    np.savetxt('../AlzData/K.csv', K, delimiter=',')
    Ks = []
    Ks.append(K)
    K2 = np.dot(K, K)
    Ks.append(K2)
    Ks.append(np.dot(K2, K))
    np.save('../AlzData/Ks', Ks)

def phenoCovariance_Alz():
    path = '/home/haohanw/AlzheimerPreprocess/generateData/'
    text = [line.strip() for line in open(path + 'traits.csv')]
    y = []
    for line in text:
        l = []
        items = line.split(',')
        for item in items:
            try:
                l.append(float(item))
            except:
                l.append(0)
        y.append(l)
    y = np.array(y)

    K = np.dot(y, y.T)
    Ks = []
    Ks.append(K)
    K2 = np.dot(K, K)
    Ks.append(K2)
    Ks.append(np.dot(K2, K))
    np.save('../AlzData/yKs', Ks)


def visualizeCovariance():
    from matplotlib import pyplot as plt
    Ks = np.load('../CancerData/Ks.npy')
    for K in Ks:
        K = rescale(K)
        print K
        plt.imshow(K)
        plt.show()


def visualizeEigenValue():
    from matplotlib import pyplot as plt
    S = np.loadtxt('../CancerData/Kva.csv', delimiter=',')
    x = np.array(xrange(S.shape[0]))
    plt.scatter(x[:-1], rescale(S[:-1]), color='y', marker='+')
    plt.scatter(x[:-1], rescale(np.power(S[:-1], 2)), color='b', marker='+')
    plt.scatter(x[:-1], rescale(np.power(S[:-1], 3)), color='m', marker='+')
    plt.show()

def visualizeYCovEigen():
    from matplotlib import pyplot as plt
    Ks = np.load('../CancerData/yKs.npy')
    for K in Ks:
        print K
        K = rescale(K)
        plt.imshow(K)
        plt.show()
    S, Kve = np.linalg.eigh(Ks[0])
    x = np.array(xrange(S.shape[0]))
    plt.scatter(x[:-1], rescale(S[:-1]), color='y', marker='+')
    plt.scatter(x[:-1], rescale(np.power(S[:-1], 2)), color='b', marker='+')
    plt.scatter(x[:-1], rescale(np.power(S[:-1], 3)), color='m', marker='+')
    plt.show()

def visualize():
    visualizeCovariance()
    visualizeEigenValue()
    visualizeYCovEigen()


if __name__ == '__main__':
    splitIntoChroms()