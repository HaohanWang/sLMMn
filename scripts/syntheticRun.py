__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
def rescale(x):
    maxi = np.max(np.abs(x))
    if maxi == 0:
        return x
    return x/maxi

def roc(beta, beta_true):
    beta = beta.flatten()
    beta = abs(beta)
    beta = rescale(beta)
    beta_true[beta_true != 0] = 1
    beta_true = beta_true.flatten()
    fpr, tpr, f = roc_curve(beta_true, beta)
    fp_prc, tp_prc, f_prc=precision_recall_curve(beta_true,beta)
    roc_auc = auc(fpr, tpr)
    return roc_auc,fp_prc,tp_prc,fpr,tpr

def run(model, seed, n, p, g, d, k, sigX, sigY,we,g_num,str1):
    from utility.syntheticDataGeneration import generateData
    from sLMMn.sLMMn import train
    np.random.seed(seed)
    discoverNum = 50
    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5
    if str1=='2':
        mode = 'lmm2'
    elif str1=='3':
        mode = 'lmm3'
    else:
        mode='lmmn'
    flag=False
    if model=='tree':
        flag=True

    snps, Y, Kva, Kve, beta_true = generateData(n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we,g_num=g_num,tree=flag,str1=str1)

    K = np.dot(snps, snps.T)
    B = []

    beta_model_lmm=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, model=model,mode=mode)
    score_model_lmm,fp_prc1,tp_prc1,fpr1,tpr1 = roc(beta_model_lmm, beta_true)
    print "score--->",score_model_lmm,'lmmn-group'
    B.append(beta_model_lmm)

    beta_model_lmm=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, model=model,mode="lmm")
    score_model_lmm,fp_prc1,tp_prc1,fpr1,tpr1 = roc(beta_model_lmm, beta_true)
    print "score--->",score_model_lmm,'lmm-g'
    B.append(beta_model_lmm)


    beta_model_linear=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum,model=model, mode="linear")
    score_model_linear,fp_prc3,tp_prc3,fpr3,tpr3 = roc(beta_model_linear, beta_true)
    print "score--->",score_model_linear,'group'
    B.append(beta_model_linear)

    beta_lasso_lmm=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum,model="lasso", mode=mode)
    score_lasso_lmm ,fp_prc2,tp_prc2,fpr2,tpr2= roc(beta_lasso_lmm, beta_true)
    print "score--->",score_lasso_lmm,'lmmn-lasso'
    B.append(beta_lasso_lmm)

    beta_lasso_lmm=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum,model="lasso", mode='lmm')
    score_lasso_lmm ,fp_prc2,tp_prc2,fpr2,tpr2= roc(beta_lasso_lmm, beta_true)
    print "score--->",score_lasso_lmm,'lmm-lasso'
    B.append(beta_lasso_lmm)

    if model == 'tree':
        fileHead = '../result/synthetic/tree/'
    else:
        fileHead = '../result/synthetic/group/'

    fileHead=fileHead+str1+'_'+str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(sigX) + '_' + str(sigY) + '_' +str(we)+'_'+str(g_num)+ '_'+str(seed) + '_'
    print fileHead
    np.save(fileHead + 'X', snps)
    np.save(fileHead + 'Y', Y)
    np.save(fileHead + 'beta1', beta_true)
    np.save(fileHead + 'beta2', B)



def syntheticRunning(cl,str1):
    from utility.syntheticDataGeneration import generateData
    from sLMMn.sLMMn import run_synthetic
    # roc = True
    # rss = []
    # for i in range(5):
    #     print '======================'
    #     print 'seed:', i
    #     generateData(i)
    #     try:
    #         for j in range(4):
    #             run_synthetic(j)
    #         rs = []
    #         for j in range(4):
    #             r = evaluateSynthetic(j, roc)
    #             rs.extend(r)
    #         rss.append(rs)
    #     except:
    #         print 'Error'
    #     print '======================'
    # print rss
    # np.savetxt('Syn_meta.csv', np.array(rss), delimiter=',')


    # m = sys.argv[1]
    # c = sys.argv[2]
    m='g'  #'g'
    c=cl
    #c='0'
    if m == 't':
        model = 'tree'
    elif m == 'g':
        model = 'group'
    str1=str1
    n = 100
    p = 500
    d = 0.05
    g = 10
    k = 50
    sigX = 0.001
    sigY=0.1
    we=0.05
    g_num = 3
    # if model=='tree':
    #     print "====================== ",c," ======================="
    #     for seed in range(5):
    #         if c == '0':
    #             run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c == 'n':
    #             for n in [300 ,1000]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c == 'p':
    #             for p in [1000, 5000]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c == 'd':
    #             for d in [0.01, 0.1]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c == 'g':
    #             for g in [5, 20]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c == 'k':  # g   useles
    #             for k in [ 20, 100]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         #here must be changed
    #         if c == 's':
    #             for sigX in [0.0005,  0.002]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c == 'c':
    #             for sigY in [10, 100]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    #         if c=='we':
    #             for we in [0.01,0.1]:
    #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    if model=='group':
        for seed in [0,1,2,3,4]:
            print "============= ",c," ==========="
            if c == '0':
                run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 'n':
                    for n in [50 ,500]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 'p':
                for p in [200,800]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 'gn':
                for g_num in [ 2,5]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 'g':
                    for g in [5, 20]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 'k':  # g   useles
                for k in [ 20, 100]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 's':
                for sigX in [0.0005,0.01]:#[0.0001,  0.01]:  #0.0005
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c == 'c':
                for sigY in [0.05,1]:#[0.01, 1]: #0.05
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
            if c=='we':
                for we in [0.01,0.1]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)


if __name__ == '__main__':
    for str1 in ['2','3','n']:
        syntheticRunning(cl='p',str1=str1)
        syntheticRunning(cl='0',str1=str1)
        syntheticRunning(cl='n',str1=str1)
        syntheticRunning(cl='p',str1=str1)
        syntheticRunning(cl='k',str1=str1)
        syntheticRunning(cl='g',str1=str1)

        syntheticRunning(cl='s',str1=str1)
        syntheticRunning(cl='c',str1=str1)
        syntheticRunning(cl='we',str1=str1)
        syntheticRunning(cl='gn',str1=str1)


