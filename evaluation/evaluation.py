#from matplotlib import pyplot as plt

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc

from utility import dataLoader

def limitPrediction(l, num):
    s = sorted(l)
    t = s[-(num+1)]
    r = []
    for v in l:
        if v > t:
            r.append(v-t)
        else:
            r.append(0)
    return r


def evaluateAT_bolt_case_select(n,roc,seed,pos):
    Y, causal = dataLoader.load_data_AT_pheno(n, seed)
    if n < 3:
        n = str(n)
    else:
        n = 'n'
    seed = str(seed)
    l = []
    for test in ['single', 'lasso']:
        # print '+++++++++++++'
        if test == 'single':
            for K in ['_bolt', '_case', '_select']:
                # print '-------------'
                sig = np.loadtxt('../ATData/K' + n + '/' + test + K + '_' + seed + '.csv', delimiter=',')
                sig = 1 - np.array(sig)
                sig = np.nan_to_num(sig)
                sigd = limitPrediction(sig, 100)
                if roc:
                    fpr, tpr = gwas_roc(sigd, causal[:, 0], positions=pos)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re = gwas_precision_recall(sigd, causal[:, 0], positions=pos)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
        else:
            for K in ['_bolt', '_case', '_select']:
                # print '-------------'
                bw = np.loadtxt('../ATData/K' + n + '/' + test + K + '_' + seed + '.csv', delimiter=',')
                bw = np.abs(bw)
                bw = limitPrediction(bw, 100)
                if roc:
                    fpr, tpr = gwas_roc(bw, causal[:, 0], positions=pos)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re = gwas_precision_recall(bw, causal[:, 0], positions=pos)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
    return l

def evaluateAT(n, roc, seed, pos):
    Y, causal = dataLoader.load_data_AT_pheno(n, seed)
    if n < 3:
        n = str(n)
    else:
        n = 'n'
    seed = str(seed)
    l = []
    for test in ['single', 'lasso']:
        # print '+++++++++++++'
        if test == 'single':
            for K in ['_linear' ,'_lmm', '_lmm2', '_lmmn']:
                # print '-------------'
                sig = np.loadtxt('../ATData/K'+n+'/'+test+K+'_'+seed+'.csv', delimiter=',')
                sig = 1-np.array(sig)
                sig = np.nan_to_num(sig)
                sigd = limitPrediction(sig, 100)
                if roc:
                    fpr, tpr = gwas_roc(sigd, causal[:,0], positions=pos)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re = gwas_precision_recall(sigd, causal[:,0], positions=pos)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
        else:
            for K in ['_linear' ,'_lmm', '_lmm2', '_lmmn']:
                # print '-------------'
                bw = np.loadtxt('../ATData/K'+n+'/'+test+K+'_'+seed+'.csv', delimiter=',')
                bw = np.abs(bw)
                bw = limitPrediction(bw, 100)
                if roc:
                    fpr, tpr = gwas_roc(bw, causal[:,0], positions=pos)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re = gwas_precision_recall(bw, causal[:,0], positions=pos)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
    return l

def evaluateSynthetic(n, roc):
    snps, Y, Kva, Kve, causal = dataLoader.load_data_synthetic(n)

    if n < 3:
        n = str(n)
    else:
        n = 'n'

    label = np.zeros(snps.shape[1])
    label[causal[:,0].astype(int)] = 1

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

def evaluateToy(n, roc):
    snps, Y, Kva, Kve, causal = dataLoader.load_data_toy(n)

    if n < 3:
        n = str(n)
    else:
        n = 'n'

    label = np.zeros(snps.shape[1])
    label[causal[:,0].astype(int)] = 1

    l = []

    for test in ['single', 'lasso']:
        # print '+++++++++++++'
        if test == 'single':
            for K in ['_linear' ,'_lmm', '_lmm2', '_lmmn']:
                # print '-------------'
                sig = np.loadtxt('../toyData/K'+n+'/'+test+K+'.csv', delimiter=',')
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
                bw = np.loadtxt('../toyData/K'+n+'/'+test+K+'.csv', delimiter=',')
                bw = np.abs(bw)
                bw = limitPrediction(bw, 50)
                if roc:
                    fpr, tpr, f = roc_curve(label, bw)
                    # print auc(fpr, tpr)
                    l.append(auc(fpr, tpr))
                else:
                    pr, re, f = precision_recall_curve(label, bw)
                    # print auc(re, pr)
                    l.append(auc(re, pr))
    return l


def getPositions(l):
    text = [line.strip() for line in open('../ATData/athaliana2.snps.chromPositionInfo.txt')][1]
    # print 'This position information is only for AT'
    pos = text.split()[:l]
    pos = [int(k) for k in pos]
    return pos

def getNearbyIndex(k, positions, nearby):
    k = int(k)
    mini = k
    maxi = k
    pos = positions[k]
    while mini>=1 and abs(positions[mini] - pos) < nearby:
        mini -=1
    l = len(positions)
    while maxi<l-2 and abs(positions[maxi] - pos) < nearby:
        maxi += 1
    return mini, maxi

def gwas_precision_recall(weights, causal_snps, positions=None, nearby=1e5):
    score = np.array(weights)
    label = np.zeros(len(weights))
    if positions is None:
        positions = getPositions(len(score))
    for k in causal_snps:
        mini, maxi = getNearbyIndex(k, positions, nearby)
        i = np.argmax(score[mini:maxi])
        label[mini+i] = 1

    p, r, t = precision_recall_curve(label, score)
    return p, r


def gwas_mse(snps, learnt_weights, causal_snps, y):
    y_pred = np.dot(snps, np.array(learnt_weights))

    causal_weights = np.zeros(snps.shape[1])
    for i in range(causal_snps.shape[0]):
        sid = causal_snps[i, 0]
        sw = causal_snps[i, 1]
        causal_weights[sid] = sw
    y_ideal = np.dot(snps, np.array(causal_weights))

    mp = np.mean(np.sum(np.square(y_pred-y)))
    mi = np.mean(np.sum(np.square(y_ideal-y)))

    return mp, mi

def gwas_nonZero(learnt_weights):
    ind = np.where(learnt_weights!=0)
    return len(ind[0])

def gwas_roc(weights, causal_snps, positions=None, nearby=1e5):
    score = np.array(weights)
    label = np.zeros(len(weights))
    if positions is None:
        positions = getPositions(len(score))
    for k in causal_snps:
        mini, maxi = getNearbyIndex(k, positions, nearby)
        i = np.argmax(score[mini:maxi])
        label[mini+i] = 1
    fpr, tpr, t = roc_curve(label, score)

    return fpr, tpr
