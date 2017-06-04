__author__ = 'Haohan Wang'

import numpy as np
import scipy
# from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import scipy
from simpleFunctions import *
import time
class TreeNode:
    def __init__(self, k):
        self.node = k
        self.leftChild = None
        self.rightChild = None
        self.parent = None

class Tree:
    def __init__(self, kList):
        if len(kList)>0:
            self.root = TreeNode(kList[0])
            left = []
            right = []
            for i in range(1, len(kList)):
                if np.random.random() > 0.5:
                    left.append(kList[i])
                else:
                    right.append(kList[i])
            self.root.leftChild = Tree(left).root
            if self.root.leftChild is not None:
                self.root.leftChild.parent = self.root
            self.root.rightChild = Tree(right).root
            if self.root.rightChild is not None:
                self.root.rightChild.parent = self.root
        else:
            self.root = None

def generateLeftChildIndex(idx, p):
    l = idx.tolist()
    c = 0
    m = len(l)
    for i in range(p):
        if i not in idx:
            l.append(i)
            c += 1
            if c >= m:
                break
    return np.array(sorted(set(l)))

def generateRightChildIndex(idx):
    l = idx.tolist()
    return np.array(l[:-len(l)/2])

def generateTreeBeta(p, k, dense):
    beta = np.zeros([p, k])
    mask = np.zeros([p, k])
    indices = [[] for i in range(k)]
    # print indices
    tree = Tree(xrange(k))
    idx = np.array(sorted((scipy.random.randint(0,p,int(p*dense)).astype(int)).tolist()))
    # print idx
    queue = []
    indices[tree.root.node] = idx
    queue.append((tree.root.leftChild, 1))
    queue.append((tree.root.rightChild, 0))
    while len(queue)>0:
        c, f = queue[0]
        queue = queue[1:]
        pind = indices[c.parent.node]
        if f == 1:
            indices[c.node] = generateLeftChildIndex(pind, p)
            # print indices[c.node]
            # print c.node
        else:
            indices[c.node] = generateRightChildIndex(pind)
            # print indices[c.node]
            # print c.node
        if c.leftChild is not None:
            queue.append((c.leftChild, 1))
        if c.rightChild is not None:
            queue.append((c.rightChild, 0))
    for i in range(len(indices)):
        mask[indices[i], i] = np.ones([indices[i].shape[0]])
        # print i," ",indices[i].shape[0]," ",np.ones([indices[i].shape[0]])

    for i in range(p):
        beta[i,:] = np.random.normal(0, 1, size=k) +i + 15
    #     for i in range(50):
    #         beta[i,:] = np.random.normal(0, 1, size=k) +i + 5
    #     for i in range(50,p):
    #         beta[i,:] = np.random.normal(0, 1, size=k) +50+ 5

    #     for i in range(p/10):
    #         beta[i,:] = np.random.normal(0, 1, size=k) +i + 5
    #     for i in range(p/10,p):
    #         beta[i,:] = np.random.normal(0, 1, size=k) +i /10+ 5+p/100*9
    #
    beta = beta*mask

    return beta


def group_beta_generation(y_number, all_number, g_num=3):

    beta_matrix = np.zeros((all_number, y_number))
    group_num = g_num

    group_size = []
    traits_left = y_number
    for i in range(group_num - 1):
        group_size.append(int(np.random.normal(traits_left / (group_num - i), 0.5, 1)))
        traits_left -= group_size[-1]
    group_size.append(traits_left)

    # For each group
    snp = 0
    traits = 0
    for i in range(group_num):
        snp_size = int(np.random.normal(all_number * 0.09, 1, 1))
        center = 500 + np.random.uniform(-3, 3, 1)
        for j in range(snp, snp + snp_size):
            beta_matrix[j][traits: traits + group_size[i]] = center
        snp += snp_size
        traits += group_size[i]

    # Between groups 0 & 1
    snp_size = int(np.random.normal(all_number * 0.015, 1, 1))
    center = 500 + np.random.uniform(-3, 3, 1)
    for j in range(snp, snp + snp_size):
        beta_matrix[j][0: group_size[0] + group_size[1]] = center * np.ones([1, group_size[0] + group_size[1]])
    snp += snp_size

    # Between groups 1 & 2
    # snp_size = int(np.random.normal(y_number / 1.5, 1, 1))
    # center = np.random.uniform(5, 7, 1)
    # for j in range(snp, snp + snp_size):
    #     beta_matrix[j][group_size[0]: group_size[0] + group_size[1] + group_size[2]] = center * np.ones([1, group_size[1] + group_size[2]])

    # Swap traits
    for i in range(int(5 * y_number)):
        a = np.random.randint(0, y_number, 1)
        b = np.random.randint(0, y_number, 1)
        beta_matrix[0:all_number, a], beta_matrix[0:all_number, b] = beta_matrix[0:all_number, b], beta_matrix[
                                                                                                   0:all_number, a]

    # Swap SNPs
    for i in range(int(5 * all_number)):
        a = np.random.randint(0, all_number, 1)
        b = np.random.randint(0, all_number, 1)
        beta_matrix[a], beta_matrix[b] = beta_matrix[b], beta_matrix[a]

    return beta_matrix  # beta_where,beta_where2,

def generateData(n, p, g, d, k, sigX, sigY,we,g_num,tree=True,str1=''):
    time_start=time.time()
    print "generate"
    dense = d
    g_num=g_num
    n = n
    p = p
    k = k
    g = g
    sig = sigX
    sigC = sigY
    we=we

    center = np.random.uniform(0, 1, [g,p])
    sample = n/g
    X = []

    for i in range(g):
        x = np.random.multivariate_normal(center[i,:], sig*np.diag(np.ones([p,])), size=sample)
        X.extend(x)
    X = np.array(X)
    #print X.shape

    # X[X>-1] = 1
    # X[X<=-1] = 0
    if tree:
        #beta=return_tree_beta(k,p)
        beta = generateTreeBeta(p, k, dense)
        beta_tmp=beta.copy()
        beta_tmp[beta_tmp != 0.] = -1
        beta_tmp[beta_tmp != -1] = 1
        beta_tmp[beta_tmp == -1] = 0
        print beta.shape
        print beta_tmp.sum()
    else:
        beta=group_beta_generation(k,p,g_num)#(int(p*dense),p)
        beta_tmp=beta.copy()
        beta_tmp[beta_tmp != 0.] = -1
        beta_tmp[beta_tmp != -1] = 1
        beta_tmp[beta_tmp == -1] = 0
        print beta.shape
        print beta_tmp.sum()
    # featureNum = int(p * dense)
    # idx = scipy.random.randint(0,p,featureNum).astype(int)
    # idx = sorted(idx)
    # w = 1*np.random.normal(0, 1, size=featureNum)
    # ypheno = scipy.dot(X[:,idx],w)
    ypheno=X.dot(beta)
    #ypheno = (ypheno-ypheno.mean())/ypheno.std()
    # ypheno = ypheno.reshape(ypheno.shape[0])
    error = np.random.normal(0, 1, ypheno.shape)

    C = np.dot(X, X.T)
    # C = centralize(C)

    Kva, Kve = np.linalg.eigh(C)
    # np.savetxt('../syntheticData/Kva.csv', Kva, delimiter=',')
    # np.savetxt('../syntheticData/Kve.csv', Kve, delimiter=',')
    # np.savetxt('../syntheticData/X.csv', X, delimiter=',')
    # causal = np.array(zip(idx, w))
    # np.savetxt('../syntheticData/causal.csv', causal, '%5.2f', delimiter=',')

    #y = we*error + normalize(ypheno)
    #np.savetxt('../syntheticData/K0/y.csv', y, '%5.2f',delimiter=',')

    yK=[]
    for i in range(ypheno.shape[1]):
        yK_ = np.random.multivariate_normal(ypheno[:, i], sigC * C, size=1)
        yK.extend(yK_)
    yK = np.array(yK)
    yK = yK.T
    yK = we * error + yK
    #np.savetxt('../syntheticData/K1/y.csv', yK1, '%5.2f',delimiter=',')

    yK2=[]
    C2 = np.dot(C, C)
    for i in range(ypheno.shape[1]):
        yK2_ = np.random.multivariate_normal(ypheno[:, i], sigC * C2, size=1)
        yK2.extend(yK2_)
    yK2 = np.array(yK2)
    yK2 = yK2.T
    yK2 = we * error + yK2

    #np.savetxt('../syntheticData/K2/y.csv', yK2, '%5.2f',delimiter=',')

    #n = np.random.randint(3, 5)
    #print n
    C3=np.dot(C2,C)

    n=3
    C_t=C
    for i in range(n):
        C = np.dot(C, C_t)

    # C3=np.dot(C,C3)
    # yKn = np.random.multivariate_normal(ypheno, sigC*C, size=1)
    # yKn = yKn.reshape(yKn.shape[1])
    # yKn = we*error + normalize(yKn)
    # yKn=[]
    # C2 = np.dot(C, C)
    yK3=[]
    for i in range(ypheno.shape[1]):
        yK3_ = np.random.multivariate_normal(ypheno[:, i], sigC * C3, size=1)
        yK3.extend(yK3_)
    yK3 = np.array(yK3)
    yK3 = yK3.T
    yK3 = we * error + yK3

    yKn=[]
    for i in range(ypheno.shape[1]):
        yKn_ = np.random.multivariate_normal(ypheno[:, i], sigC * C, size=1)
        yKn.extend(yKn_)
    yKn = np.array(yKn)
    yKn = yKn.T
    yKn = we * error + yKn


    #np.savetxt('../syntheticData/Kn/y.csv', yKn, '%5.2f',delimiter=',')
    time_end = time.time()
    time_diff = time_end - time_start
    print '%.2fs to generate data' % (time_diff)

    if str1=='2':
        return X,yK2,Kva, Kve,beta
    elif str1=='3':
            return X,yK3,Kva, Kve,beta
    else:
        return X,yKn,Kva, Kve,beta
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