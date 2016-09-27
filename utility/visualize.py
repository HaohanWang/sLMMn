__author__ = 'Haohan Wang'

from matplotlib import pyplot as plt
import numpy as np

from dataLoader import load_data_synthetic

def normalize(m):
    s = np.sum(m)
    return m/s

def visualize(m):
    normalize(m)
    plt.imshow(m)
    plt.show()

snps, Y, Kva, Kve, causal = load_data_synthetic(1)
K1 = normalize(np.dot(snps, snps.T))

S, U = np.linalg.eigh(K1)
S = normalize(S)
print S
# print normalize(np.power(S, 2))
print normalize(S + normalize(np.power(S, 2)))

# K2 = np.dot(K1, K1)
# K3 = np.dot(K2, K1)
# K4 = np.dot(K3, K1)
#
# visualize(K1)
# visualize(K2)
# visualize(K1+K2)
# visualize(K3)
# visualize(K1+K2+K3)