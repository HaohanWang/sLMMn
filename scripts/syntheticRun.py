__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from utility import dataLoader
from evaluation.evaluation import evaluateSynthetic

def syntheticRunning():
    from utility.syntheticDataGeneration import generateData
    from sLMMn.sLMMn import run_synthetic
    roc = True
    rss = []
    for i in range(20):
        print '======================'
        print 'seed:', i
        generateData(i)
        try:
            for j in range(4):
                run_synthetic(j)
            rs = []
            for j in range(4):
                r = evaluateSynthetic(j, roc)
                rs.extend(r)
            rss.append(rs)
        except:
            print 'Error'
        print '======================'
    print rss
    np.savetxt('Syn_meta.csv', np.array(rss), delimiter=',')

def syntheticSingleRun(seed):
    from utility.syntheticDataGeneration import generateData
    from sLMMn.sLMMn import run_synthetic
    roc = True
    generateData(seed)
    for j in range(4):
        run_synthetic(j)
    for j in range(4):
        r = evaluateSynthetic(j, roc)
        print r


if __name__ == '__main__':
    syntheticRunning()
