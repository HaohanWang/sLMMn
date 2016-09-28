__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np

def ATRunning():
    from utility.ATDataGeneration import generateData
    from sLMMn.sLMMn import run_AT
    from evaluation.evaluation import evaluateAT
    roc = True
    rss = []
    for i in range(10):
        print '======================'
        print 'seed:', i
        generateData(i)
        for j in range(4):
            run_AT(j, i)
        rs = []
        for j in range(4):
            r = evaluateAT(j, roc, i)
            rs.extend(r)
        rss.append(rs)
        print '======================'
    print rss
    np.savetxt('AT_meta.csv', np.array(rss), delimiter=',')

def ATRunningSingle(seed):
    from utility.ATDataGeneration import generateData
    from sLMMn.sLMMn import run_AT
    generateData(seed)
    for j in range(4):
        run_AT(j, seed)

def ATEvaluation():
    from evaluation.evaluation import evaluateAT
    roc = True
    rss = []
    for i in range(5):
        rs = []
        for j in range(4):
            r = evaluateAT(j, roc, i)
            rs.extend(r)
        rss.append(rs)
        print rs
        print '======================'
    print rss
    np.savetxt('AT_meta.csv', np.array(rss), delimiter=',')


if __name__ == '__main__':
    # seed = int(sys.argv[1])
    # ATRunningSingle(seed)
    ATEvaluation()