__author__ = 'Haohan Wang'

import numpy as np

class ConjugateGradientMethod:
    def __init__(self, maxIter=30, tol=1e-3):
        self.maxIter = maxIter
        self.tol = tol

    def solve(self, A, b):
        '''
        :param A: n, n
        :param b: n, 1
        :return:
        '''
        b=b.reshape(b.shape[0],1)
        x = np.zeros([A.shape[1], 1]) # n, 1
        temp=np.dot(A,x)
        temp=temp.reshape(temp.shape[0],1)
        r = b - temp # n, 1
        p = r # n, 1
        rsold = np.dot(r.T, r)  #1, 1
        for iter in range(self.maxIter):
            Ap = np.dot(A, p)   #
            alpha = rsold / np.dot(p.T, Ap)
            x = x +  alpha*p
            r = r - alpha * Ap
            rsnew = np.dot(r.T, r)
            if rsnew < self.tol**2:
                return x
            p = r + (rsnew/rsold)*p
            rsold = rsnew
        return x

