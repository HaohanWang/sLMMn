__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import scipy.optimize as opt

from helpingMethods import *


class lmm_select:
    def __init__(self, numintervals=100, ldeltamin=-5, ldeltamax=5, mode='lmm', alpha=0.05, fdr=False,
                 threshold=0.5):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.mode = mode




    def fit(self, X, K, Kva, Kve, y, mode='linear'):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))
        X0 = np.ones(len(y)).reshape(len(y), 1)
        SUX = X
        SUy = y
        SUX0 = None
        SUy=SUy.reshape(SUy.shape[0],1)

        w1 = self.hypothesisTest(SUX, SUy, X, SUX0, X0)
        self.beta = np.array(w1)
        return self.beta


    def fit2(self, X, K, Kva, Kve, y, mode):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))

        X0 = np.ones(len(y)).reshape(len(y), 1)


        if mode != 'linear':
            S, U, ldelta0 = self.train_nullmodel(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                             ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)

            delta0 = scipy.exp(ldelta0)
            Sdi = 1. / (S + delta0)
            Sdi_sqrt = scipy.sqrt(Sdi)
            SUX = scipy.dot(U.T, X)
            SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
            SUy = scipy.dot(U.T, y)
            SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
            SUX0 = scipy.dot(U.T, X0)
            SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
        else:
            SUX = X
            SUy = y
            ldelta0 = 0
            monitor_nm = {}
            monitor_nm['ldeltaopt'] = 0
            monitor_nm['nllopt'] = 0
            SUX0 = None

        w1 = self.hypothesisTest(SUX, SUy, X, SUX0, X0)
        w1=np.array(w1)
        w1[w1==0]=1e-7
        self.beta = -np.log(w1)
        #self.beta[self.beta<=(-np.log(0.05))]==0
        return self.beta


    def hypothesisTest(self,UX, Uy, X, UX0, X0):
        [m, n] = X.shape
        p = []
        for i in range(n):
            if UX0 is not None:
                UXi = np.hstack([UX0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(UXi.T, UXi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
                #print beta.shape,
                Uyr = Uy - matrixMult(UXi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            else:
                Xi = np.hstack([X0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(Xi.T, Xi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
                Uyr = Uy - matrixMult(Xi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            # print "~~~~~~~~~~~~~~~~~~~~~"
            ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
            if -1e10 < ts < 1e10:
                p.append(ps)
            else:
                p.append(1)
            #print p,
        return p



    def train_nullmodel(self, y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm', p=1):
        ldeltamin += scale
        ldeltamax += scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = scipy.dot(U.T, y)

        # grid search
        nllgrid = scipy.ones(numintervals + 1) * scipy.inf
        ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        for i in scipy.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S) # the method is in helpingMethods

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
        return S, U, ldeltaopt_glob