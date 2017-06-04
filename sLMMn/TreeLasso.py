__author__ = "Xiang Liu"

#from numpy import random
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt


class treeNode:
    def __init__(self, trait=[], children=[], s=0., weight=0.):
        self.trait = trait
        self.children = children
        self.s = s
        self.weight = weight


class minXY:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Tree:
    def __init__(self, root=None):
        self.root = root

    def getRoot(self):
        return self.root

    def setRoot(self, root=treeNode()):
        self.root = root

    def setWeight(self):
        n = self.root.s
        prev_s = 0.
        last = 1
        first = 0
        nodes = [self.root]
        while (first != last):
            node = nodes[first]
            prev_s = float(node.s)
            node.s = float(node.s / n)
            for i in range(0, len(node.children)):
                node.children[i].s = prev_s - 1
                nodes.append(node.children[i])
                last = last + 1
            first = first + 1

    def buildParentFromChildren(self, chd):
        par = treeNode()
        par.trait = []
        for i in range(0, len(chd)):
            par.trait.extend(chd[i].trait)
            if ((chd[i].s + 1) > par.s):
                par.s = chd[i].s + 1.
        par.children = chd
        return par

    def buildLeafNode(self, t):
        n = treeNode()
        n.trait = [t]
        n.s = 1.
        return n


class TreeLasso:
    def __init__(self, lambda_=0., clusteringMethod="single", threhold=1., mu=0.01, T=Tree(), initGradientFlag=False,
                 mD_=None, XX=None, XY=None, mTw=None, mT=None, C=None, gIdx=None, tauNorm=None, L=None, X=None, y=None,
                 beta=None, mD=None, maxEigen=None):
        self.lambda_ = lambda_
        self.clusteringMethod = clusteringMethod
        self.threhold = threhold
        self.mu = mu
        self.T = T
        self.initGradientFlag = initGradientFlag
        self.X = X
        self.y = y
        self.beta = beta
        self.L = L
        self.tauNorm = tauNorm
        self.gIdx = gIdx
        self.mD_ = mD_
        self.XX = XX
        self.XY = XY
        self.mTw = mTw
        self.mT = mT
        self.C = C
        self.mD = mD
        self.maxEigen = maxEigen

    def getBeta(self):
        return self.beta

    def setXY(self, m, n):
        self.X = m
        self.y = n
        self.initBeta()

    def setLambda(self, d):
        self.lambda_ = d

    def hierarchicalClustering(self):
        n = self.y.shape[1]
        weights = np.zeros((n, n))
        maps = {}
        self.T = Tree()
        for i in range(0, n):
            t = self.T.buildLeafNode(i)
            maps[i] = t
            temp1 = self.y[:, i]
            temp1 = np.array(temp1)
            temp1 = temp1.reshape(temp1.shape[0], 1)
            temp = self.y - temp1
            for j in range(0, temp.shape[1]):
                weights[i, j] = np.linalg.norm(temp[:, j], 2)
                weights[i, j] = weights[i, j] * weights[i, j]
        xy = minXY()
        while (len(maps) > 1):
            xy = self.searchMin(weights)
            t1 = maps[xy.x]
            t2 = maps[xy.y]
            tv = []
            tv.append(t1)
            tv.append(t2)
            p = self.T.buildParentFromChildren(tv)
            maps[len(maps)] = p
            weights = self.appendColRow(weights, xy)
            weights = self.removeColRow(weights, xy)
            self.updateMap(maps, xy)
        root = maps[0]
        self.T.setRoot(root)
        self.setWeight()

    def removeColRow(self, mptr, xy):
        mptr = self.removeRow(mptr, xy.y)
        mptr = self.removeRow(mptr, xy.x)
        mptr = self.removeCol(mptr, xy.y)
        mptr = self.removeCol(mptr, xy.x)
        return mptr

    def removeRow(self, mptr, x):

        numRows = mptr.shape[0] - 1
        numCols = mptr.shape[1]
        if x < numRows:
            mptr = np.delete(mptr, x, 0)
        return mptr

    def removeCol(self, mptr, y):
        numCols = mptr.shape[1] - 1
        numRows = mptr.shape[0]
        if y < numCols:
            mptr = np.delete(mptr, y, 1)
        return mptr

    def updateMap(self, mptr, xy):
        n = len(mptr)
        del mptr[xy.x]
        del mptr[xy.y]
        for i in range(xy.x + 1, n):
            if i != xy.y:
                t = mptr[i]
                if i < xy.y:
                    k = i - 1
                elif i > xy.y:
                    k = i - 2
                del mptr[i]
                mptr[k] = t

    def appendColRow(self, mat, xy):
        r = mat.shape[0]
        result = np.zeros((r + 1, r + 1))
        col = np.zeros(r)
        if self.clusteringMethod == "average":
            col = (mat[:, xy.x] + mat[:, xy.y]) / 2
        elif self.clusteringMethod == "complete":
            col = np.array([mat[:, xy.x], mat[:, xy.y]]).max(axis=0)
        else:
            col = np.array([mat[:, xy.x], mat[:, xy.y]]).min(axis=0)
        result[0:r, 0:r] = mat
        col = col.reshape(col.shape[0], 1)
        result[0:r, r:r + 1] = col
        result[r:r + 1, 0:r] = col.T
        return result

    def searchMin(self, m):
        r = m.shape[0]
        xy = minXY()
        tmpV = np.inf
        for i in range(0, r):
            for j in range(i + 1, r):
                if m[i, j] < tmpV:
                    tmpV = m[i, j]
                    xy.x = i
                    xy.y = j
        return xy

    def setClusteringMethod(self, str):
        self.clusteringMethod = str

    def setThrehold(self, thred):
        self.threhold = thred

    def initBeta(self):
        c = self.X.shape[1]
        d = self.y.shape[1]
        #random.seed(0)
        self.beta=np.zeros((c,d))
        # self.beta = random.random(size=(c, d))
        # self.beta = 2 * self.beta - 1
        # self.beta=np.zeros((c,d))
        # self.beta=np.loadtxt('/home/miss-iris/Code/ConfoundingCorrection/data/data_group_tree/TL/beta.csv', delimiter=',')
        # print self.beta

    def setWeight(self):
        self.T.setWeight();
        self.penaltyWeights();
        self.prune();
        self.initMatrixD();

    def getTree(self):
        return self.T

    def cost(self):
        return ((self.y - self.X.dot(self.beta)) * (self.y - self.X.dot(self.beta))).sum() / 2 + self.penalty_cost() * 5

    def penalty_cost(self):
        self.initGradientUpdate()
        A = self.C.dot(self.beta.T)
        c = A.shape[1]
        s = 0.
        v = self.gIdx.shape[0]
        for i in range(0, v):
            tmp = np.zeros(c)
            for j in range(int(self.gIdx[i, 0]) - 1, int(self.gIdx[i, 1])):
                tmp += np.square(A[j, :])
            tmp = np.sqrt(tmp)
            s += tmp.sum()
        return s

    def l2NormIndex(self, traits):
        r = 0.
        for i in range(0, len(traits)):
            r = r + np.linalg.norm(self.beta[:, traits[i]])
        return r

    def l2NormIndexIndex(self, j, traits):
        r = 0.
        for i in range(0, len(traits)):
            r = r + np.linalg.norm(self.beta[j, traits[i]])
        return r

    def prune(self):
        nodes = []
        nodes = [self.T.getRoot()]
        first = 0
        last = 1
        while (first != last):
            n = nodes[first]
            if n.s > self.threhold:
                n.weight = 0.
            if len(n.children) > 0:
                for i in range(0, len(n.children)):
                    nodes.append(n.children[i])
                    last = last + 1
            first = first + 1

    def penaltyWeights(self):
        nodes = []
        root = self.T.getRoot()
        first = 0
        last = 1
        nodes = [root]
        while (first != last):
            n = nodes[first]
            if len(n.children) != 0:
                for i in range(0, len(n.children)):
                    n.children[i].weight = n.s * (1. - n.children[i].s)
                    nodes.append(n.children[i])
                    last = last + 1
            first = first + 1

    def countNodes(self):
        nodes = []
        count = 1
        first = 0
        last = 1
        nodes = [self.T.getRoot()]
        while (first != last):
            n = nodes[first]
            if len(n.children) != 0:
                for i in range(0, len(n.children)):
                    nodes.append(n.children[i])
                    last = last + 1
                    count = count + 1
            first = first + 1
        return count

    def countNoneZeroNodes(self):
        nodes = []
        count = 0
        first = 0
        last = 1
        nodes = [self.T.getRoot()]
        while (first != last):
            n = nodes[first]
            if len(n.children) != 0:
                for i in range(0, len(n.children)):
                    nodes.append(n.children[i])
                    last = last + 1
            if n.weight != 0:
                count = count + 1
            first = first + 1
        return count

    def initMatrixD(self):
        c = self.X.shape[1]
        d = self.countNodes()
        self.mD = np.zeros((c, d))
        self.mD_ = np.zeros((c, d))

    def updateMD(self):
        index = 0
        c = self.X.shape[1]
        nodes = [self.T.getRoot()]
        first = 0
        last = 1
        denominator = self.updateMD_denominator()
        while (first != last):
            n = nodes[first]
            for j in range(0, c):
                self.mD_[j, index] = denominator * n.weight / self.l2NormIndexIndex(j, n.trait)
            if len(n.children) != 0:
                for i in range(0, len(n.children)):
                    nodes.append(n.children[i])
                    last = last + 1
            first = first + 1
            index = index + 1

    def updateMD_denominator(self):
        r = 0.
        nodes = [self.T.getRoot()]
        first = 0
        last = 1
        while (first != last):
            n = nodes[first]
            r += n.weight * self.l2NormIndex(n.trait)
            if len(n.children) != 0:
                for i in range(0, len(n.children)):
                    nodes.append(n.children[i])
                    last = last + 1
            first = first + 1
        return r

    def updateBeta(self):  # todo: there are two updateBetas
        k = self.beta.shape[1]
        n = self.XX.shape[0]
        D = np.zeros((n, n))
        for i in range(0, n):
            D[i, i] = self.mD_[i:, ].sum()
        xxdx = np.linlg.inv(self.XX + self.lambda_ * D).dot(self.X.T)
        for j in range(0, k):
            self.beta[:, j] = xxdx.dot(self.y[:, j])

    def updateBeta(self, b):
        self.beta = b

    def setMu(self, m):
        self.mu = m

    def getL(self):
        return self.L

    def proximal_operator(self, in_, l):
        sign = in_.copy()
        sign[sign > 0] = 1
        sign[sign < 0] = -1
        in_ = abs(in_) - l * self.lambda_ / self.L
        in_[in_ < 0] = 0
        return in_ * sign

    def proximal_derivative(self):
        A = self.C.dot(self.beta.T) / self.mu
        r = A.shape[0]
        c = A.shape[1]
        R = np.zeros((r, c))
        v = self.gIdx.shape[0]
        for i in range(0, v):
            tmp = np.zeros(c)
            for j in range(int(self.gIdx[i, 0]) - 1, int(self.gIdx[i, 1])):
                tmp = tmp + np.square(A[j, :])
            tmp = np.sqrt(tmp)
            for j in range(0, tmp.shape[0]):
                if tmp[j] < 1:
                    tmp[j] = 1
            for j in range(int(self.gIdx[i, 0]) - 1, int(self.gIdx[i, 1])):
                R[j, :] = A[j, :] / tmp.T
        return self.X.T.dot((self.X.dot(self.beta))) - self.XY + R.T.dot(self.C)

    def initGradientUpdate(self):
        if not self.initGradientFlag:
            self.initGradientFlag = True
            nodeNum = self.countNoneZeroNodes()
            c = len(self.T.getRoot().trait)
            r = nodeNum - c
            self.mT = np.zeros((r, c))
            self.mTw = np.zeros((r, 1))
            self.gIdx = np.zeros((r, 3))
            self.gIdx[0, 0] = 1
            index = r - 1
            nodes = [self.T.getRoot()]
            first = 0
            last = 1
            Cweights = []
            Cindex = []
            last1 = 0
            last2 = 0
            while (first != last):
                node = nodes[first]
                if len(node.children) > 0:
                    if (node.weight != 0):
                        self.mTw[index, 0] = node.weight
                        for j in range(0, len(node.trait)):
                            self.mT[index, node.trait[j]] = 1
                            Cweights.append(node.weight)
                            Cindex.append(node.trait[j])
                            last1 = last1 + 1
                            last2 = last2 + 1
                        index = index - 1
                    for i in range(0, len(node.children)):
                        nodes.append(node.children[i])
                        last = last + 1
                first = first + 1
            self.C = np.zeros((len(Cweights), c))
            tmpIndex = 0
            while (last2 != 0):
                self.C[tmpIndex, Cindex[last2 - 1]] = Cweights[last1 - 1]
                last2 = last2 - 1
                last1 = last1 - 1
                tmpIndex = tmpIndex + 1
            for i in range(0, r):
                s = int(self.mT[i, :].sum())
                self.gIdx[i, 1] = self.gIdx[i, 0] + s - 1
                self.gIdx[i, 2] = s
                if i + 1 < r:
                    self.gIdx[i + 1, 0] = self.gIdx[i, 1] + 1
            tau = np.zeros(c)
            for i in range(0, r):
                tau = tau + self.mT[i, :] * (self.mTw[i, 0] * self.mTw[i, 0])
            self.tauNorm = tau.max()
            # L1, L2_ = np.linalg.eigh(self.X.T.dot(self.X))
            if self.maxEigen is None:
                s= np.linalg.svd(self.X, full_matrices=False)[1]
                L1 = np.max(s)
                L1 = L1*L1
            else:
                L1 = self.maxEigen
            self.L = L1 + self.lambda_ * self.lambda_ * self.tauNorm / self.mu
            self.XY = self.X.T.dot(self.y)

if __name__ == '__main__':
    pass
