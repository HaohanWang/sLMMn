__author__ = 'Haohan Wang'

import scipy.optimize as opt
import time
from sklearn.linear_model import Lasso

import sys

sys.path.append('../')

from helpingMethods import *

from utility import dataLoader

def train(X, K, Kva, Kve, y, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, mode='linear'):
    """
    train linear mixed model lasso

    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    rho: augmented Lagrangian parameter for Lasso solver
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8) for Lasso solver

    Output:
    results
    """
    time_start = time.time()
    [n_s, n_f] = X.shape
    assert X.shape[0] == y.shape[0], 'dimensions do not match'
    assert K.shape[0] == K.shape[1], 'dimensions do not match'
    assert K.shape[0] == X.shape[0], 'dimensions do not match'
    if y.ndim == 1:
        y = scipy.reshape(y, (n_s, 1))

    X0 = np.ones(len(y)).reshape(len(y), 1)

    if mode != 'linear':
        S, U, ldelta0, monitor_nm = train_nullmodel(y, K, S=Kva, U=Kve, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode)

        delta0 = scipy.exp(ldelta0)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, X)
        SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        SUy = scipy.dot(U.T, y)
        TransY = scipy.reshape(Sdi_sqrt, (n_s, 1))
        # SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1)) # todo: here the fastLMM and lmm-lasso are different
        SUX0 = scipy.dot(U.T, X0)
        SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
    else:
        SUX = X
        SUy = y
        ldelta0 = 0
        monitor_nm = {}
        monitor_nm['ldeltaopt'] = 0
        monitor_nm['nllopt'] = 0
        TransY = None
        SUX0 = None

    w1 = hypothesisTest(SUX, SUy, X, TransY, SUX0, X0)
    regs = []
    for i in range(5, 20):
        for j in range(1, 10):
            regs.append(j*np.power(10.0, -i))
    breg, ss = cv_train(SUX, SUy.reshape([n_s, 1]), transY=TransY,regList=regs, SKlearn=True, selectK=True, K=discoverNum)
    w2 = train_lasso(SUX, SUy, breg, TransY)

    time_end = time.time()
    time_diff = time_end - time_start
    print '... finished in %.2fs' % (time_diff)

    res = {}
    res['ldelta0'] = ldelta0
    res['single'] = w1
    res['combine'] = w2
    res['combine_ss'] = ss
    res['combine_reg'] = regs
    res['time'] = time_diff
    res['monitor_nm'] = monitor_nm
    return res


def train_lasso(X, y, mu, TransY):
    if TransY is not None:
        y = y * TransY
    lasso = Lasso(alpha=mu)
    lasso.fit(X, y)
    return lasso.coef_

def hypothesisTest(UX, Uy, X, TransY, UX0, X0):
    [m, n] = X.shape
    p = []
    for i in range(n):
        if TransY is not None:
            UXi = np.hstack([UX0 ,UX[:, i].reshape(m, 1)])
            Xi = np.hstack([X0 ,X[:, i].reshape(m, 1)])
            XX = matrixMult(UXi.T, Xi)
            XX_i = linalg.pinv(XX)
            beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
            Uyr = Uy - matrixMult(Xi, beta)
            Q = np.dot( (Uyr* TransY).T, Uyr)
            sigma = Q * 1.0 / m
        else:
            Xi = np.hstack([X0 ,X[:, i].reshape(m, 1)])
            XX = matrixMult(Xi.T, Xi)
            XX_i = linalg.pinv(XX)
            beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
            Uyr = Uy - matrixMult(Xi, beta)
            Q = np.dot(Uyr.T, Uyr)
            sigma = Q * 1.0 / m
        ts, ps = tstat(beta[1], np.abs(XX_i[1, 1]), sigma, 1, m)
        if -1e10 < ts < 1e10:
            p.append(ps)
    return p

def nLLeval(ldelta, Uy, S, REML=True):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = scipy.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    if REML:
        pass

    return nLL


def train_nullmodel(y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm'):
    """
    train random effects model:
    min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,

    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    """
    ldeltamin += scale
    ldeltamax += scale

    if S is None or U is None:
        S, U = linalg.eigh(K)

    if mode == 'lmm2':
        S = np.power(S, 2) + S

    Uy = scipy.dot(U.T, y)

    # grid search
    if mode != 'lmmn':
        nllgrid = scipy.ones(numintervals + 1) * scipy.inf
        ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        for i in scipy.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)

        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        for i in scipy.arange(numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        monitor = {}
        monitor['ldeltaopt'] = ldeltaopt_glob
        monitor['nllopt'] = nllmin
    else:
        Stmp = S
        kchoices = [1, 2, 3, 4]
        knum = len(kchoices)
        global_S = S
        global_ldeltaopt = scipy.inf
        global_min = scipy.inf
        for ki in range(knum):
            kc = kchoices[ki]
            if kc == 1:
                Stmp = S
            else:
                Stmp += np.power(S, kc)
            Uy = scipy.dot(U.T, y)
            nllgrid = scipy.ones(numintervals + 1) * scipy.inf
            ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
            nllmin = scipy.inf
            for i in scipy.arange(numintervals + 1):
                nllgrid[i] = nLLeval(ldeltagrid[i], Uy, Stmp)
            nll_min = nllgrid.min()
            ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]
            for i in scipy.arange(numintervals - 1) + 1:
                if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                    ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, Stmp),
                                                                  (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                                  full_output=True)
                    if nllopt < nllmin:
                        nll_min = nllopt
                        ldeltaopt_glob = ldeltaopt
            # print kc, nll_min, ldeltaopt_glob
            if nll_min < global_min:
                global_min = nll_min
                global_ldeltaopt = ldeltaopt_glob
                global_S = np.copy(S)
        ldeltaopt_glob = global_ldeltaopt
        S = global_S
        monitor = {}
        monitor['nllopt'] = global_min
        monitor['ldeltaopt'] = ldeltaopt_glob

    return S, U, ldeltaopt_glob, monitor


def cv_train(X, Y, transY, regList, SKlearn=True, selectK=False, K=100):
    if transY is not None:
        Y = Y*transY
    ss = []
    b = np.inf
    breg = 0
    for reg in regList:
        clf = Lasso(alpha=reg)
        clf.fit(X, Y)
        k = len(np.where(clf.coef_ != 0)[0])
        s = np.abs(k - K)
        ss.append(s)
        if s < b:
            b = s
            breg = reg
    return breg, ss

def run_synthetic(dataMode):
    discoverNum = 100
    snps, Y, Kva, Kve, causal = dataLoader.load_data_synthetic(dataMode)
    K = np.dot(snps, snps.T)
    if dataMode < 3:
        dataMode = str(dataMode)
    else:
        dataMode = 'n'
    for mode in ['linear', 'lmm', 'lmm2', 'lmmn']:
        res = train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, mode=mode)
        # hypothesis weights
        fileName1 = '../syntheticData/K'+dataMode+'/single_' + mode
        result = np.array(res['single'])
        ldelta0 = res['ldelta0']
        np.savetxt(fileName1 + '.csv', result, delimiter=',')
        f2 = open(fileName1 + '.hmax.txt', 'w')
        f2.writelines(str(ldelta0)+'\n')
        f2.close()
        # lasso weights
        bw = res['combine']
        regs = res['combine_reg']
        ss = res['combine_ss']
        fileName2 = '../syntheticData/K'+dataMode+'/lasso_' + mode
        f1 = open(fileName2 + '.csv', 'w')
        for wi in bw:
            f1.writelines(str(wi) + '\n')
        f1.close()
        f0 = open(fileName2 + '.regularizerScore.txt', 'w')
        for (ri, si) in zip(regs, ss):
            f0.writelines(str(ri) + '\t' + str(si) + '\n')
        f0.close()


if __name__ == '__main__':
    at = int(sys.argv[1])
    snp = sys.argv[2]
    pass
