__author__ = 'Haohan Wang'

import numpy as np
import scipy
# from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

def centralize(x):
    m = np.mean(x)
    return x-m

def normalize(x):
    m = np.mean(x)
    s = np.std(x)
    return (x-m)/s

def generateData(seed):
    np.random.seed(seed)

    dense = 0.05

    n = 1000
    p = 100
    g = 5
    sig = 1
    sigC = 1

    we = 1

    center = np.random.uniform(0, 1, [g,p])
    sample = n/g
    X = []

    for i in range(g):
        x = np.random.multivariate_normal(center[i,:], sig*np.diag(np.ones([p,])), size=sample)
        X.extend(x)
    X = np.array(X)
    print X.shape

    X[X>-1] = 1
    X[X<=-1] = 0

    featureNum = int(p * dense)
    idx = scipy.random.randint(0,p,featureNum).astype(int)
    idx = sorted(idx)
    w = 1*np.random.normal(0, 1, size=featureNum)
    ypheno = scipy.dot(X[:,idx],w)
    ypheno = (ypheno-ypheno.mean())/ypheno.std()
    ypheno = ypheno.reshape(ypheno.shape[0])
    error = np.random.normal(0, 1, n)

    C = np.dot(X, X.T)
    # C = centralize(C)

    Kva, Kve = np.linalg.eigh(C)
    np.savetxt('../syntheticData/Kva.csv', Kva, delimiter=',')
    np.savetxt('../syntheticData/Kve.csv', Kve, delimiter=',')
    np.savetxt('../syntheticData/X.csv', X, delimiter=',')
    causal = np.array(zip(idx, w))
    np.savetxt('../syntheticData/causal.csv', causal, '%5.2f', delimiter=',')

    y = we*error + normalize(ypheno)
    np.savetxt('../syntheticData/K0/y.csv', y, '%5.2f',delimiter=',')

    yK1 = np.random.multivariate_normal(ypheno, sigC*C, size=1)
    yK1 = yK1.reshape(yK1.shape[1])
    yK1 = we*error + normalize(yK1)
    np.savetxt('../syntheticData/K1/y.csv', yK1, '%5.2f',delimiter=',')

    C2 = np.dot(C, C)
    yK2 = np.random.multivariate_normal(ypheno, sigC*C2, size=1)
    yK2 = yK2.reshape(yK2.shape[1])
    yK2 = we*error + normalize(yK2)
    np.savetxt('../syntheticData/K2/y.csv', yK2, '%5.2f',delimiter=',')

    n = np.random.randint(1, 4)
    for i in range(n):
        C = np.dot(C, C)
    yKn = np.random.multivariate_normal(ypheno, sigC*C, size=1)
    yKn = yKn.reshape(yKn.shape[1])
    yKn = we*error + normalize(yKn)
    np.savetxt('../syntheticData/Kn/y.csv', yKn, '%5.2f',delimiter=',')

    # x = xrange(len(y))
    # plt.scatter(x, y, color='g')
    # plt.scatter(x, yK1, color='r')
    # plt.scatter(x, yK2, color='b')
    # plt.scatter(x, yKn, color='m')
    # plt.show()
    # print normalize(y)
    # print normalize(yK1)
    # print normalize(yK2)
    # print normalize(yKn)

    # Z = linkage(X, 'ward')
    #
    # from scipy.cluster.hierarchy import cophenet
    # from scipy.spatial.distance import pdist
    #
    # c, coph_dists = cophenet(Z, pdist(X))
    #
    # plt.figure(figsize=(25, 10))
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')
    # dendrogram(
    #     Z,
    #     leaf_rotation=90.,  # rotates the x axis labels
    #     leaf_font_size=8.,  # font size for the x axis labels
    # )
    # plt.show()

if __name__=='__main__':
    generateData(1)