__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from utility import dataLoader
from evaluation.evaluation import evaluateToy

def toyRunning():
    from utility.toyDataGeneration import generateData
    from sLMMn.sLMMn import run_toy
    roc = True
    rss = []
    for i in range(20):
        print '======================'
        print 'seed:', i
        generateData(i)
        try:
            for j in range(4):
                run_toy(j)
            rs = []
            for j in range(4):
                r = evaluateToy(j, roc)
                rs.extend(r)
            rss.append(rs)
        except:
            print 'Error'
        print '======================'
    print rss
    np.savetxt('Toy_meta2.csv', np.array(rss), delimiter=',')

def toySingleRun(seed):
    from utility.toyDataGeneration import generateData
    from sLMMn.sLMMn import run_toy
    roc = True
    generateData(seed)
    for j in range(4):
        run_toy(j)
    for j in range(4):
        r = evaluateToy(j, roc)
        print r


if __name__ == '__main__':
    toySingleRun(0)
