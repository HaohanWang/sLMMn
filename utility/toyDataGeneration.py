__author__ = 'Haohan Wang'

import numpy as np
import scipy
# from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans


def centralize(x):
    m = np.mean(x)
    return x - m

def mapping2ZeroOne(x):
    maxi = np.max(x)
    mini = np.min(x)
    return (x-mini)/(maxi-mini)

def normalize(x):
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / s

def generateData(seed, test=False):
    plt = None
    if test:
        from matplotlib import pyplot as plt
    np.random.seed(seed)

    dense = 0.05

    n = 100
    p = 1000
    zp = 5
    g = 5
    sig = 1
    sigC = 1

    we = 0.01

    center = np.random.uniform(0, 1, [g, p])
    center2 = np.random.uniform(0, 1, [g, zp])
    sample = n / g
    X = []
    Z = []

    for i in range(g):
        x = np.random.multivariate_normal(center[i, :], sig * np.diag(np.ones([p, ])), size=sample)
        X.extend(x)
        for j in range(sample):
            Z.append(center2[i])
    X = np.array(X)
    Z = np.array(Z)
    print X.shape

    X[X > -1] = 1
    X[X <= -1] = 0

    featureNum = int(p * dense)
    idx = scipy.random.randint(0, p, featureNum).astype(int)
    idx = sorted(idx)
    w = 1 * np.random.normal(0, 1, size=featureNum)
    ypheno = scipy.dot(X[:, idx], w)
    ypheno = (ypheno - ypheno.mean()) / ypheno.std()
    ypheno = ypheno.reshape(ypheno.shape[0])
    error = np.random.normal(0, 1, n)

    C = np.dot(Z, Z.T)
    if test:
        plt.imshow(C)
        plt.show()

    Kva, Kve = np.linalg.eigh(C)
    if test:
        ind = xrange(Kva.shape[0])
        plt.scatter(ind, mapping2ZeroOne(Kva), color='r', marker='+')
        plt.scatter(ind, mapping2ZeroOne(np.power(Kva, 2)), color='b', marker='+')
        plt.scatter(ind, mapping2ZeroOne(np.power(Kva, 3)), color='m', marker='+')
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

    yK1 = np.random.multivariate_normal(ypheno, sigC * C, size=1)
    yK1 = yK1.reshape(yK1.shape[1])
    yK1 = we * error + normalize(yK1)
    if not test:
        np.savetxt('../toyData/K1/y.csv', yK1, '%5.2f', delimiter=',')

    C2 = np.dot(C, C)
    yK2 = np.random.multivariate_normal(ypheno, sigC * C2, size=1)
    yK2 = yK2.reshape(yK2.shape[1])
    yK2 = we * error + normalize(yK2)
    if test:
        plt.imshow(C2)
        plt.show()
    if not test:
        np.savetxt('../toyData/K2/y.csv', yK2, '%5.2f', delimiter=',')

    n = np.random.randint(3, 4)
    Ct = C
    for i in range(n):
        C = np.dot(Ct, C)
    if test:
        plt.imshow(C)
        plt.show()
    yKn = np.random.multivariate_normal(ypheno, sigC * C, size=1)
    yKn = yKn.reshape(yKn.shape[1])
    yKn = we * error + normalize(yKn)
    if not test:
        np.savetxt('../toyData/Kn/y.csv', yKn, '%5.2f', delimiter=',')

    if test:
        x = xrange(len(y))
        plt.scatter(x, y, color='g')
        plt.scatter(x, yK1, color='r')
        plt.scatter(x, yK2, color='b')
        plt.scatter(x, yKn, color='m')
        plt.show()


if __name__ == '__main__':
    generateData(0, test=True)
