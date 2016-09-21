__author__ = 'Haohan Wang'

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

np.random.seed(12)

dense = 0.05

n = 100
p = 1000
g = 5
sig = 1

we = 0.01
wh = 1
wg = [0.5, 0.25, 0.125, 0.0625]
# wg = [1, 1, 1, 1]

center = np.random.uniform(0, 1, [g,p])
sample = n/g
X = []

for i in range(g):
    x = np.random.multivariate_normal(center[i,:], sig*np.diag(np.ones([p,])), size=sample)
    # plt.scatter(x[:,0], x[:,1], c=c[i])
    X.extend(x)
X = np.array(X)
print X.shape

X[X>-1] = 1
X[X<=-1] = 0

# plt.show()

featureNum = int(p * dense)
confoundNum = int(n*0.5)
idx = scipy.random.randint(0,p,featureNum).astype(int)
idx = sorted(idx)
w = 1*np.random.normal(0, 1, size=featureNum)
ypheno = scipy.dot(X[:,idx],w)
ypheno = (ypheno-ypheno.mean())/ypheno.std()
ypheno = ypheno.reshape(ypheno.shape[0])
error = np.random.normal(0, 1, n)

cl = KMeans(n_clusters=g)
m = cl.fit_predict(X)
c = cl.cluster_centers_

Z = c[m,:]

# print Z.shape

idx_z = scipy.random.randint(0,n,confoundNum).astype(int)
idx_z = sorted(idx_z)
w_z = 1*np.random.normal(0, 1, size=confoundNum)
ygroup = scipy.dot(Z[:,idx_z],w_z)
ygroup = (ygroup-ygroup.mean())/ygroup.std()
ygroup = ygroup.reshape(ygroup.shape[0])

# print ygroup
C = np.dot(X, X.T)

Kva, Kve = np.linalg.eigh(C)
np.savetxt('../syntheticData/Kva.csv', Kva, delimiter=',')
np.savetxt('../syntheticData/Kve.csv', Kve, delimiter=',')
np.savetxt('../syntheticData/X.csv', X, delimiter=',')
causal = np.array(zip(idx, w))
np.savetxt('../syntheticData/causal.csv', causal, '%5.2f', delimiter=',')
ys = []

for i in range(3):
    idx_c = scipy.random.randint(0,n,confoundNum).astype(int)
    w_c = 1*np.random.normal(0, 1, size=confoundNum)
    m = cl.fit_predict(C)
    c = cl.cluster_centers_

    yc = scipy.dot(Z[:,idx_c],w_c)
    yc = (yc-yc.mean())/yc.std()
    yc = yc.reshape(yc.shape[0])
    ys.append(yc)
    C = np.dot(C,C)

y = we*error + wh*ypheno
np.savetxt('../syntheticData/K0/y.csv', y, '%5.2f',delimiter=',')

y+= wg[0]*ygroup
np.savetxt('../syntheticData/K1/y.csv', y, '%5.2f',delimiter=',')

y += wg[1]*ys[1]
np.savetxt('../syntheticData/K2/y.csv', y, '%5.2f',delimiter=',')
for i in range(1, 3):
    y += wg[i+1]*ys[i]

np.savetxt('../syntheticData/Kn/y.csv', y, '%5.2f',delimiter=',')

x = xrange(len(y))
plt.scatter(x, y)
plt.show()

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
