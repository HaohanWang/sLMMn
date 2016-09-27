__author__ = 'Haohan Wang'

import numpy as np

path = '/home/haohanw/FaSTLMM_K2_Sparsity/data'

def calculateHigherOrderKinship():
    snps = np.loadtxt(path + '/athaliana2.snps.csv', delimiter=',')
    C = np.dot(snps, snps.T)
    C2 = np.dot(C, C)
    C3 = np.dot(C2, C)
    C4 = np.dot(C3, C)

    np.savetxt('C1.csv', C, delimiter=',')
    np.savetxt('C2.csv', C2, delimiter=',')
    np.savetxt('C3.csv', C3, delimiter=',')
    np.savetxt('C4.csv', C4, delimiter=',')

    phenos = np.loadtxt(path + '/athaliana2.phenos.csv', delimiter=',')

    P = np.dot(phenos, phenos.T)

    np.savetxt('C.csv', P, delimiter=',')


if __name__ == '__main__':
    calculateHigherOrderKinship()
