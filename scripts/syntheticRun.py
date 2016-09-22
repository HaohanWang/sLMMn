__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from utility import dataLoader
from evaluation.evaluationAUC import limitPrediction
from sLMMn.sLMMn import run_synthetic

def evaluateSynthetic(n):
    snps, Y, Kva, Kve, causal = dataLoader.load_data_synthetic(n)

    if n < 3:
        n = str(n)
    else:
        n = 'n'

    label = np.zeros(snps.shape[1])
    label[causal[0,:].astype(int)] = 1

    l = []

    for test in ['single', 'lasso']:
        # print '+++++++++++++'
        if test == 'single':
            for K in ['_linear' ,'_lmm', '_lmm2', '_lmmn']:
                # print '-------------'
                sig = np.loadtxt('../syntheticData/K'+n+'/'+test+K+'.csv', delimiter=',')
                sig = 1-np.array(sig)
                sig = np.nan_to_num(sig)
                sigd = limitPrediction(sig, 50)
                if roc:
                    fpr, tpr, f = roc_curve(label, sigd)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re, f = precision_recall_curve(label, sigd)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
        else:
            for K in ['_linear' ,'_lmm', '_lmm2', '_lmmn']:
                # print '-------------'
                bw = np.loadtxt('../syntheticData/K'+n+'/'+test+K+'.csv', delimiter=',')
                bw = np.abs(bw)
                if roc:
                    fpr, tpr, f = roc_curve(label, bw)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re, f = precision_recall_curve(label, bw)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
    return l

if __name__ == '__main__':
    np.random.seed(0)
    from utility.syntheticDataGeneration import generateData
    roc = True
    rss = []
    for i in range(20):
        print '======================'
        print 'seed:', i
        generateData(i)
        for i in range(4):
            run_synthetic(i)
        rs = []
        for i in range(4):
            r = evaluateSynthetic(i)
            rs.extend(i)
        rss.append(rs)
        print '======================'
    np.savetxt('meta.csv', np.array(rss), delimiter=',')
