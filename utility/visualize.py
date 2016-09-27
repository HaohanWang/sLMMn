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

C = np.loadtxt('../tmpData/C.csv', delimiter=',')
C1 = np.loadtxt('../tmpData/C1.csv', delimiter=',')
C2 = np.loadtxt('../tmpData/C2.csv', delimiter=',')
C3 = np.loadtxt('../tmpData/C3.csv', delimiter=',')
C4 = np.loadtxt('../tmpData/C4.csv', delimiter=',')
print C

visualize(C)
visualize(C1)
visualize(C2)
visualize(C3)
visualize(C4)