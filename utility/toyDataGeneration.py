__author__ = 'Haohan Wang'

import scipy
# from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

from utility.simpleFunctions import *


def generateData(seed, test=False):
    plt = None
    if test:
        from matplotlib import pyplot as plt
    np.random.seed(seed)

    dense = 0.05

    n = 100
    p1 = 900
    p2 = 100
    g = 20
    sig = 1
    sigC = 1e5
    p = 1000

    we = 0.01

    center1 = np.random.normal(0, 1, [g, p1])
    center2 = np.random.normal(0, 1, [g, p2])
    sample = n / g
    X = []

    for i in range(g):
        x1 = np.random.multivariate_normal(center1[i, :], sig * np.diag(np.ones([p1, ])), size=sample)
        x2 = np.tile(center2[i,:], [sample, 1])
        x = np.append(x1, x2, 1)
        X.extend(x)
    X = np.array(X)

    X[X > -1] = 1
    X[X <= -1] = 0
    Z = X

    if test:
        plt.imshow(X)
        plt.show()

    featureNum = int(p * dense)
    idx = scipy.random.randint(0, p, featureNum).astype(int)
    idx = sorted(idx)
    w = 1 * np.random.normal(0, 1, size=featureNum)
    ypheno = scipy.dot(X[:, idx], w)
    ypheno = (ypheno - ypheno.mean()) / ypheno.std()
    ypheno = ypheno.reshape(ypheno.shape[0])
    error = np.random.normal(0, 1, n)

    C = np.dot(Z, Z.T)
    C1 = rescale(C)
    if test:
        plt.imshow(C1)
        plt.show()
        print C1

    Kva, Kve = np.linalg.eigh(C)
    if test:
        ind = np.array(xrange(Kva.shape[0]))
        plt.scatter(ind[:-1], mapping2ZeroOne(Kva[:-1]), color='y', marker='+')
        plt.scatter(ind[:-1], mapping2ZeroOne(np.power(Kva, 2)[:-1]), color='b', marker='+')
        plt.scatter(ind[:-1], mapping2ZeroOne(np.power(Kva, 4)[:-1]), color='m', marker='+')
        plt.show()
    if not test:
        np.savetxt('../toyData/Kva.csv', Kva, delimiter=',')
        np.savetxt('../toyData/Kve.csv', Kve, delimiter=',')
        np.savetxt('../toyData/X.csv', X, delimiter=',')
    causal = np.array(zip(idx, w))
    if not test:
        np.savetxt('../toyData/causal.csv', causal, '%5.2f', delimiter=',')

    y = we * error + normalize(ypheno)
    if not test:
        np.savetxt('../toyData/K0/y.csv', y, '%5.2f', delimiter=',')

    yK1 = np.random.multivariate_normal(ypheno, sigC * C1, size=1)
    yK1 = yK1.reshape(yK1.shape[1])
    yK1 = we * error + normalize(yK1)
    if not test:
        np.savetxt('../toyData/K1/y.csv', yK1, '%5.2f', delimiter=',')

    C2 = np.dot(C, C)
    C2 = rescale(C2)
    yK2 = np.random.multivariate_normal(ypheno, sigC * C2, size=1)
    yK2 = yK2.reshape(yK2.shape[1])
    yK2 = we * error + normalize(yK2)
    if test:
        plt.imshow(C2)
        plt.show()
        print C2
    if not test:
        np.savetxt('../toyData/K2/y.csv', yK2, '%5.2f', delimiter=',')

    C3 = np.dot(C2,C)
    C3 = rescale(C3)
    if test:
        plt.imshow(C3)
        plt.show()
        print C3
    yKn = np.random.multivariate_normal(ypheno, sigC * C3, size=1)
    yKn = yKn.reshape(yKn.shape[1])
    yKn = we * error + normalize(yKn)
    if not test:
        np.savetxt('../toyData/Kn/y.csv', yKn, '%5.2f', delimiter=',')

    if test:
        x = xrange(len(y))
        plt.scatter(x, y, color='g')
        plt.scatter(x, yK1, color='y')
        plt.scatter(x, yK2, color='b')
        plt.scatter(x, yKn, color='m')
        plt.show()

        def imshowY(y):
            y = y.reshape(y.shape[0], 1)
            plt.imshow(np.dot(y, y.T))
            plt.show()

        imshowY(y)
        imshowY(yK1)
        imshowY(yK2)
        imshowY(yKn)


if __name__ == '__main__':
    generateData(2, test=True)
