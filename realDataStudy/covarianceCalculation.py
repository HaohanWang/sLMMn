__author__ = 'Haohan Wang'

from utility.simpleFunctions import *

def calculateCovariance():
    path = '/home/haohanw/FaSTLMM_K2_Sparsity/data/'
    X = np.loadtxt(path + 'athaliana.snps.chrom1.csv', delimiter=',')
    K = np.dot(X, X.T)
    Kva, Kve = np.linalg.eigh(K)
    np.savetxt('../ATData/Kva.csv', Kva, delimiter=',')
    np.savetxt('../ATData/Kve.csv', Kve, delimiter=',')
    np.savetxt('../ATData/K.csv', K, delimiter=',')
    Ks = []
    Ks.append(K)
    K2 = np.dot(K,K)
    Ks.append(K2)
    Ks.append(np.dot(K2, K))
    np.save('../ATData/Ks', Ks)

def visualizeCovariance():
    from matplotlib import pyplot as plt
    Ks = np.load('../ATData/Ks.npy')
    for K in Ks:
        K = rescale(K)
        print K
        plt.imshow(K)
        plt.show()


if __name__ == '__main__':
    calculateCovariance()