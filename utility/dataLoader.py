import numpy as np
from scipy import linalg

path = '/home/haohanw/FaSTLMM_K2_Sparsity/data/'

def load_data_synthetic(n):
    if n == 3:
        n = 'n'
    else:
        n = str(n)
    snps = np.loadtxt('../syntheticData/X.csv', delimiter=',')
    Y = np.loadtxt('../syntheticData/K'+n+'/y.csv', delimiter=',')
    causal = np.loadtxt('../syntheticData/causal.csv', delimiter=',')
    Kva = np.loadtxt('../syntheticData/Kva.csv', delimiter=',')
    Kve = np.loadtxt('../syntheticData/Kve.csv', delimiter=',')
    return snps, Y, Kva, Kve, causal


def load_data_AT_basic():
    X = np.loadtxt(path + 'athaliana.snps.chrom1.csv', delimiter=',')
    K = np.loadtxt('../ATData/K.csv', delimiter=',')
    Kva = np.loadtxt('../ATData/Kva.csv', delimiter=',')
    Kve = np.loadtxt('../ATData/Kve.csv', delimiter=',')
    return X, K, Kva, Kve

def load_data_AT_pheno(n, s):
    if n == 3:
        n = 'n'
    else:
        n = str(n)
    s = str(s)
    Y = np.loadtxt('../syntheticData/K'+n+'/y_'+s+'.csv', delimiter=',')
    causal = np.loadtxt('../syntheticData/causal_'+s+'.csv', delimiter=',')
    return Y, causal