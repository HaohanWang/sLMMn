__author__ = 'Haohan Wang'

import scipy.linalg as linalg
import scipy
import numpy as np
from scipy import stats

def matrixMult(A, B):
    try:
        linalg.blas
    except AttributeError:
        return np.dot(A, B)

    if not A.flags['F_CONTIGUOUS']:
        AA = A.T
        transA = True
    else:
        AA = A
        transA = False

    if not B.flags['F_CONTIGUOUS']:
        BB = B.T
        transB = True
    else:
        BB = B
        transB = False

    return linalg.blas.dgemm(alpha=1., a=AA, b=BB, trans_a=transA, trans_b=transB)

def factor(X, rho):
    """
    computes cholesky factorization of the kernel K = 1/rho*XX^T + I

    Input:
    X design matrix: n_s x n_f (we assume n_s << n_f)
    rho: regularizaer

    Output:
    L  lower triangular matrix
    U  upper triangular matrix
    """
    n_s, n_f = X.shape
    K = 1 / rho * scipy.dot(X, X.T) + scipy.eye(n_s)
    U = linalg.cholesky(K)
    return U

def tstat(beta, var, sigma, q, N, log=False):

    """
       Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
       This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
    """
    ts = beta / np.sqrt(var * sigma)
    # ts = beta / np.sqrt(sigma)
    # ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), self.N-q))
    # sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
    if log:
        ps = 2.0 + (stats.t.logsf(np.abs(ts), N - q))
    else:
        ps = 2.0 * (stats.t.sf(np.abs(ts), N - q))
    if not len(ts) == 1 or not len(ps) == 1:
        raise Exception("Something bad happened :(")
        # return ts, ps
    return ts.sum(), ps.sum()