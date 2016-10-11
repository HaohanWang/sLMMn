__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np

def ATRunning():
    from utility.ATDataGeneration import generateData
    from sLMMn.sLMMn import run_AT
    from evaluation.evaluation import evaluateAT
    # get position information
    text = [line.strip() for line in open('../ATData/athaliana.snps.chromPositionInfo.txt')]
    chrom = [int(k) for k in text[0].split()]
    pos_tmp = [int(k) for k in text[1].split()]
    pos = []
    for i in range(len(chrom)):
        if chrom[i] == 5:
            pos.append(pos_tmp[i])

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
            r = evaluateAT(j, roc, i, pos)
            rs.extend(r)
        rss.append(rs)
        print '======================'
    print rss
    np.savetxt('AT_meta.csv', np.array(rss), delimiter=',')

def ATRunningSingle(seed):
    from utility.ATDataGeneration import generateData
    from sLMMn.sLMMn import run_AT
    from evaluation.evaluation import evaluateAT
    generateData(seed)
    roc = True
    text = [line.strip() for line in open('../ATData/athaliana.snps.chromPositionInfo.txt')]
    chrom = [int(k) for k in text[0].split()]
    pos_tmp = [int(k) for k in text[1].split()]
    pos = []
    for i in range(len(chrom)):
        if chrom[i] == 5:
            pos.append(pos_tmp[i])

    for j in range(4):
        run_AT(j, seed)
    for j in range(4):
        r = evaluateAT(j, roc, seed, pos)
        print r

if __name__ == '__main__':
    ATRunningSingle(0)