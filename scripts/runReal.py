__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# from utility import dataLoader
# from evaluation.evaluation import evaluateSynthetic
from utility.syntheticDataGeneration import generateData
from sLMMn.sLMMn import train

def loadRealData(dt):
    if dt == 'alz':
        path = '../realData/AlzData/'
        X = np.load(path + 'snps.npy')
        y = np.load(path + 'phenoMulti.npy')
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        return X, y, Kva, Kve
    elif dt == 'at':
        path = '../realData/ATData/'
        X = np.load(path + 'geno.npy')
        y = np.load(path + 'pheno.npy')
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        ind = [0, 1, 2, 3, 4, 5, 6, 39, 40,
               32, 33, 34, 73, 74, 75, 101, 102, 98, 99, 48, 49, 50, 51, 52, 71, 72,
               82,83, 84, 65, 66, 67, 79, 80, 81, 88, 89, 90, 7, 100, 104, 105, 106, 91]
        # ind=[0,1,2]
        y = y[:, ind]
        return X, y, Kva, Kve
    # elif dt == 'mice':
    #     ind = range(49, 58) + range(78, 97)
    #     path = '/home/haohanw/miceData/'
    #     X = np.load(path + 'snps.npy')
    #     y = np.load(path + 'pheno.npy')[:, ind]
    #     Kva = np.load(path + 'Kva.npy')
    #     Kve = np.load(path + 'Kve.npy')
    #     return X, y, Kva, Kve

def run(cat,str1):

    discoverNum = 50
    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5
    model='tree'
    if str1=='2':
        mode = 'lmm2'
    else:
        mode='lmmn'
    print mode
    snps, Y, Kva, Kve = loadRealData(cat)
    # print snps.shape
    # print Y.shape
    K = np.dot(snps, snps.T)

    B = []

    beta_model_lmm_=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, model=model,mode=mode,flag=True)
    B.append(beta_model_lmm_)

    fileHead = '../result/real/tree/'
    np.save(fileHead +str1 +'_beta_'+cat, B)

if __name__ == '__main__':
    cat = 'alz'
    str1='2'
    run(cat,str1)
    str1='3'
    run(cat,str1)
