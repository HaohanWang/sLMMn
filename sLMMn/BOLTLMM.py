__author__ = 'Haohan Wang'

import numpy as np
from ConjugateGradientMethod import ConjugateGradientMethod

from helpingMethods import *


class BOLTLMM:
    def __init__(self, maxIter=10):
        self.cgm = ConjugateGradientMethod()
        self.maxIter = maxIter

    def logdelta(self, h):
        return np.log((1 - h) / h)

    def train(self, X, y,f2,p):
        self.X = X
        self.y = y
        sigma_g, sigma_e = self.estimate_variance_parameter()
        delta = sigma_e / sigma_g
        inf_stats = self.inf_statistics(delta)
        #f2, p = self.estimate_Gaussian_prior(sigma_g, sigma_e)
        ## we can just use only the f2 and p from the paper to find the biggest roc~
        f2=f2
        p=p
        # print "===",f2,p
        pvalues = self.Gaussian_mixture_statistics(sigma_g, sigma_e, f2, p, inf_stats)
        beta = -np.log(pvalues)
        beta[beta < -np.log(0.05)] = 0
        return beta

    def estimate_variance_parameter(self):
        [self.n, self.p] = self.X.shape
        secantIteration = 7  # fix parameters

        ld = np.zeros([secantIteration])
        h = np.zeros([secantIteration])
        f = np.zeros([secantIteration])
        delta = 0
        for i in range(secantIteration):
            if i == 0:
                h[i] = 0.25
                ld[i] = self.logdelta(h[i])
                f[i], delta = self.evalfREML(ld[i])
            elif i == 1:
                if h[i - 1] < 0:
                    h[i] = 0.125
                else:
                    h[i] = 0.5
                ld[i] = self.logdelta(h[i])
                f[i], delta = self.evalfREML(ld[i])
            else:
                ld[i] = (ld[i - 2] * f[i - 1] - ld[i - 1] * f[i - 2]) / (f[i - 1] - f[i - 2])
                if abs(ld[i] - ld[i - 1]) < 0.01:
                    break
                f[i], delta = self.evalfREML(ld[i])

        sigma_g = np.dot(self.y.T, self.Hy) / self.n
        sigma_e = delta * sigma_g

        return sigma_g, sigma_e


    def evalfREML(self, ldelta):
        delta = np.exp(ldelta)
        if delta>=1e6:
            delta=1e6
        H = np.eye(self.n)*delta + np.dot(self.X, self.X.T)/self.p
        self.Hy = self.cgm.solve(H, self.y)
        MCtrials = max(min(4e9/self.n**2, 15), 3) # fix parameters
        beta_hat = np.zeros([self.p, MCtrials])
        e_hat = np.zeros([self.n, MCtrials])
        for t in range(MCtrials):
            beta_rand = np.random.normal(size=[self.p, 1])
            e_rand = np.random.normal(size=[self.n, 1])
            y_rand = np.dot(self.X, beta_rand) + np.sqrt(delta) * e_rand
            Hy_rand = self.cgm.solve(H, y_rand)

            temp=np.dot(self.X.T, Hy_rand)
            temp=temp.reshape(temp.shape[0],)
            beta_hat[:, t] = temp/self.n
            temp2=delta*Hy_rand
            temp2=temp2.reshape(temp2.shape[0],)
            e_hat[:,t] = temp2

        beta_data = 1./self.n*np.dot(self.X.T, self.Hy)
        e_data = delta*self.Hy

        s_beta_hat = 1e-8
        s_e_hat = 1e-8
        for t in range(MCtrials):
            s_beta_hat += np.dot(beta_hat[:,t].T, beta_hat[:,t])
            s_e_hat += np.dot(e_hat[:,t].T, e_hat[:,t])
        s1=(MCtrials*((np.dot(beta_data.T, beta_data))/(np.dot(e_data.T, e_data)+1e-8)))
        return np.log(s1/(s_beta_hat/s_e_hat)), delta


    def inf_statistics(self, delta):
        snpNum = 30  # fix parameters
        prospectiveStat = np.zeros([snpNum])
        uncalibratedRestrospectiveStat = np.zeros([snpNum])
        V = np.eye(self.n) * delta + np.dot(self.X, self.X.T) / self.p
        for t in range(snpNum):
            ind = np.random.randint(0, self.p)
            x = self.X[:, ind]
            Vx = self.cgm.solve(V, x)
            xHy2 = np.square(np.dot(x.T, self.Hy))
            prospectiveStat[t] = xHy2 / (np.dot(x.T, Vx))
            uncalibratedRestrospectiveStat[t] = self.n * xHy2 / (np.linalg.norm(x) * np.linalg.norm(self.Hy))

        infStatCalibration = np.sum(uncalibratedRestrospectiveStat) / np.sum(prospectiveStat)

        stats = np.zeros([self.p, 1])
        for i in range(self.p):
            x = self.X[:, i]
            xHy2 = np.square(np.dot(x.T, self.Hy))
            stats[i] = (self.n * xHy2 / (np.linalg.norm(x) * np.linalg.norm(self.Hy))) / infStatCalibration
        return stats

    def estimate_Gaussian_prior(self, sigma_g, sigma_e):
        min_f2 = 0
        min_p = 0
        min_mse = np.inf
        for f2 in [0.5, 0.3, 0.1]:  # fix parameters
            for p in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:  # fix parameters
                mse = 0
                for Xtrain, ytrain, Xtest, ytest in KFold(self.X, self.y, 5):
                    beta, y_resid = self.fitVariationalBayes(Xtrain, ytrain, sigma_g, sigma_e, f2, p)
                    mse += np.linalg.norm(np.dot(Xtest, beta) - ytest)
                if mse < min_mse:
                    min_mse = mse
                    min_f2 = f2
                    min_p = p
        return min_f2, min_p

    def fitVariationalBayes(self, Xtrain, ytrain, sigma_g, sigma_e, f2, p):
        sigma_b = [sigma_g / self.n * (1 - f2) / p, sigma_e / self.p * f2 / (1 - p)]

        beta = np.zeros([self.p, 1])
        yresid = self.y
        approxLL = -np.inf

        for t in range(self.maxIter):
            approxLLprev = approxLL
            approxLL = -self.n / 2 * np.log(2 * np.pi * sigma_e + 1e-8)
            for j in range(self.p):
                beta_bar = [0, 0]
                tao = [0, 0]
                s = [0, 0]
                x = self.X[:, j]
                xnorm = np.linalg.norm(x)
                yresid += beta[j] * x
                beta_m = np.dot(x.T, yresid) / (xnorm)
                for i in range(2):
                    beta_bar[i] = beta_m * sigma_b[i] / (sigma_b[i] + sigma_e / (xnorm) + 1e-8)
                    tao[i] = (sigma_b[i] * sigma_e / xnorm) / (sigma_b[i] + sigma_e / xnorm + 1e-8)
                    s[i] = np.sqrt(sigma_b[i] + sigma_e / xnorm + 1e-8)
                ###########here the 1e-8 1e-9 1e-10 just in case that we divide 0~
                pm = (p / (s[0] + 1e-8) * np.exp(-np.square(beta_m) / (2 * np.square(s[0]) + 1e-9)) + 1e-9) / \
                     ((p / (s[0] + 1e-8) * np.exp(-np.square(beta_m) / (2 * np.square(s[0]))) + 1e-8) + (
                     1e-9 + (1 - p) / (s[1] + 1e-8) * np.exp(-np.square(beta_m) / (2 * np.square(s[1]) + 1e-8))))

                beta[j] = pm * beta_bar[0] + (1 - pm) * beta_bar[1]

                var_beta_m = pm * (tao[0] + beta_bar[0]) - np.square(pm * beta_bar[0])

                kl = pm * np.log(pm / p) + (1 - pm) * np.log((1 - pm) / (1 - p)) - pm / 2 * (
                1 + np.log((tao[0] + 1e-9) / (sigma_b[0] + 1e-8)) - (tao[0] + beta_bar[0] + 1e-10) / (
                sigma_b[0] + 1e-8))
                ###################################change
                approxLL -= xnorm / (2 * sigma_e + 1e-8) * var_beta_m + kl
                if approxLL==np.inf:
                    #print j, approxLL
                    approxLL=1e50
                    break
                elif approxLL==-np.inf :
                    #print j, approxLL
                    approxLL=-1e50
                    break
                elif approxLL>=1e50:
                    #print j, approxLL
                    approxLL=1e50
                    break
                elif approxLL<=-1e50:
                    #print j, approxLL
                    approxLL=-1e50
                    break
                ############################change

                yresid -= beta[j] * x
            approxLL = np.linalg.norm(yresid) / (2 * sigma_e + 1e-8)

            if approxLL - approxLLprev < 0.01:
                break

        return beta, yresid

    def Gaussian_mixture_statistics(self, sigma_g, sigma_e, f2, p, infstatis):
        uncalibratedBoltLmm = np.zeros([self.p, 1])
        beta, yresid = self.fitVariationalBayes(self.X, self.y, sigma_g, sigma_e, f2, p)

        for j in range(self.p):
            x = self.X[:, j]
            uncalibratedBoltLmm[j] = self.n * np.square(np.dot(x.T, yresid)) / (
            np.linalg.norm(x) * np.linalg.norm(yresid))

        # LD score
        LDscoreCalibration = 1
        return uncalibratedBoltLmm / LDscoreCalibration