__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# from utility import dataLoader
# from evaluation.evaluation import evaluateSynthetic
from utility.syntheticDataGeneration import generateData
from sLMMn.sLMMn import train

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

def run(model, seed, n, p, g, d, k, sigX, sigY,we,g_num,str1,simulate=False):

    np.random.seed(seed)
    discoverNum = 50
    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5
    # model='tree'
    if str1=='2':
        mode = 'lmm2'#lmmn
    else:
        mode='lmmn'
    print mode
    flag=False
    if model=='tree':
        flag=True

    snps, Y, Kva, Kve, beta_true = generateData(n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we,g_num=g_num,tree=flag,str1=str1)




    K = np.dot(snps, snps.T)

    B = []

    # slmm_model = sLMMn(discoverNum=discoverNum, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode,
    #                   numintervals=numintervals,
    #                   model=model)
    # beta_model_lmm = slmm_model.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    beta_model_lmm_=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, model=model,mode=mode)
    score_model_lmm_,fp_prc1,tp_prc1,fpr1,tpr1 = roc(beta_model_lmm_, beta_true)
    print 'lmm-n-tree:',score_model_lmm_
    B.append(beta_model_lmm_)

    beta_model_lmm=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, model=model,mode="lmm")
    score_model_lmm,fp_prc2,tp_prc2,fpr2,tpr2 = roc(beta_model_lmm, beta_true)
    print "lmm-tree:",score_model_lmm
    B.append(beta_model_lmm)


    # beta_model_linear = slmm_linear.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    beta_model_linear=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum,model=model, mode="linear")
    score_model_linear,fp_prc3,tp_prc3,fpr3,tpr3 = roc(beta_model_linear, beta_true)
    print "tree:",score_model_linear
    B.append(beta_model_linear)

    # beta_lasso_lmm = slmm_lasso.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    beta_lasso_lmm_=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum,model="lasso", mode=mode)
    score_lasso_lmm_ ,fp_prc4,tp_prc4,fpr4,tpr4= roc(beta_lasso_lmm_, beta_true)
    print "lmm-n-lasso:",score_lasso_lmm_
    B.append(beta_lasso_lmm_)

    beta_lasso_lmm=train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum,model="lasso", mode="lmm")
    score_lasso_lmm ,fp_prc5,tp_prc5,fpr5,tpr5= roc(beta_lasso_lmm, beta_true)
    print "lmm-lasso:",score_lasso_lmm
    B.append(beta_lasso_lmm)
    if simulate==False:
        if model == 'tree':
            fileHead = '../result/synthetic/tree/'
        else:
            fileHead = '../result/synthetic/group/'

        fileHead = fileHead + str(str1)+'_'+str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sigX) + '_' + str(sigY) + '_'+str(we)+'_' +str(g_num)+'_'+ str(seed) + '_2_'
        print fileHead

        np.save(fileHead + 'X', snps)
        np.save(fileHead + 'Y', Y)
        np.save(fileHead + 'beta1', beta_true)
        np.save(fileHead + 'beta2', B)
    else:
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        im = ax.imshow(beta_true.T)
        ax.title.set_text('Simulated mapping linkage matrix')
        plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure_new/'+str1+'simulated_beta_true.png', dpi=300)
        plt.show()

        fig = plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        im = ax.imshow(Y.T)
        ax.title.set_text('Simulated mapping linkage matrix')
        plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure_new/'+str1+'simulated_beta_true.png', dpi=300)
        plt.show()


        fig = plt.figure()
        ax=fig.add_subplot(5, 1, 1)
        im = ax.imshow(beta_model_lmm_.T)
        # print beta_true
        # print "--------------------"
        ax.title.set_text('lmm-n-tree')
        ax1 = fig.add_subplot(5, 1, 2)
        #beta_model_lmm=500*beta_model_lmm
        # print beta_model_lmm
        # print "--------------------"
        im1 = ax1.imshow(beta_model_lmm.T)
        ax1.title.set_text('lmm-tree')
        ax2 = fig.add_subplot(5, 1, 3)
        #beta_model_linear=500*beta_model_linear
        # print beta_model_linear
        # print "--------------------"
        im2 = ax2.imshow(beta_model_linear.T)
        ax2.title.set_text('Tree Lasso')
        ax3 = fig.add_subplot(5, 1, 4)
        #beta_lasso_lmm=beta_lasso_lmm
        # print beta_lasso_lmm
        # print "--------------------"
        im3 = ax3.imshow(beta_lasso_lmm_.T)
        ax3.title.set_text('lmm-n-lasso')
        ax4 = fig.add_subplot(5, 1, 5)
        #beta_lasso_lmm=beta_lasso_lmm
        # print beta_lasso_lmm
        # print "--------------------"
        im4 = ax3.imshow(beta_lasso_lmm.T)
        ax4.title.set_text('sLMM')
        # plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure_new/'+str1+'_beta_all.png', dpi=500)
        plt.show()


        fig = plt.figure()
        ax=fig.add_subplot(5, 1, 1)
        im = ax.imshow(abs(beta_model_lmm_).T)
        # print beta_true
        # print "--------------------"
        ax.title.set_text('lmm-n-tree')
        ax1 = fig.add_subplot(5, 1, 2)
        #beta_model_lmm=500*beta_model_lmm
        # print beta_model_lmm
        # print "--------------------"
        im1 = ax1.imshow(abs(beta_model_lmm.T))
        ax1.title.set_text('lmm-tree')
        ax2 = fig.add_subplot(5, 1, 3)
        #beta_model_linear=500*beta_model_linear
        # print beta_model_linear
        # print "--------------------"
        im2 = ax2.imshow(abs(beta_model_linear.T))
        ax2.title.set_text('Tree Lasso')
        ax3 = fig.add_subplot(5, 1, 4)
        #beta_lasso_lmm=beta_lasso_lmm
        # print beta_lasso_lmm
        # print "--------------------"
        im3 = ax3.imshow(abs(beta_lasso_lmm_.T))
        ax3.title.set_text('lmm-n-lasso')
        ax4 = fig.add_subplot(5, 1, 5)
        #beta_lasso_lmm=beta_lasso_lmm
        # print beta_lasso_lmm
        # print "--------------------"
        im4 = ax3.imshow(abs(beta_lasso_lmm.T))
        ax4.title.set_text('sLMM')
        # plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure_new/'+str1+'_beta_all2.png', dpi=500)
        plt.show()



        y1=snps.dot(beta_model_lmm_)
        y2=snps.dot(beta_model_lmm)
        y3=snps.dot(beta_model_linear)
        y4=snps.dot(beta_lasso_lmm_)
        y5=snps.dot(beta_lasso_lmm)

        fig = plt.figure()
        ax=fig.add_subplot(5, 1, 1)
        im = ax.imshow(y1.T)
        # print beta_true
        # print "--------------------"
        ax.title.set_text('lmm-n-tree')
        ax1 = fig.add_subplot(5, 1, 2)
        #beta_model_lmm=500*beta_model_lmm
        # print beta_model_lmm
        # print "--------------------"
        im1 = ax1.imshow(y2.T)
        ax1.title.set_text('lmm-tree')
        ax2 = fig.add_subplot(5, 1, 3)
        #beta_model_linear=500*beta_model_linear
        # print beta_model_linear
        # print "--------------------"
        im2 = ax2.imshow(y3.T)
        ax2.title.set_text('Tree Lasso')
        ax3 = fig.add_subplot(5, 1, 4)
        #beta_lasso_lmm=beta_lasso_lmm
        # print beta_lasso_lmm
        # print "--------------------"
        im3 = ax3.imshow(y4.T)
        ax3.title.set_text('lmm-n-lasso')
        ax4 = fig.add_subplot(5, 1, 5)
        #beta_lasso_lmm=beta_lasso_lmm
        # print beta_lasso_lmm
        # print "--------------------"
        im4 = ax3.imshow(y5.T)
        ax4.title.set_text('sLMM')
        # plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure_new/'+str1+'_y_all.png', dpi=500)
        plt.show()






def syntheticRunning(cl,str1,simulate=False):
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
    m='t'
    c=cl
    model=''
    if m == 't':
        model = 'tree'
    elif m == 'g':
        model = 'group'
    str1=str1
    simulate=simulate
    if simulate==False:
        n = 500
        p = 3000
        d = 0.05
        g = 10
        k = 50
        sigX = 1e-5
        sigY=0.1
        we=0.05
        g_num = 3
        if model=='tree':
            print "====================== ",c," ======================="
            for seed in range(5):
                print "~~~~~~~~~~~~",seed,"~~~~~~~~~~~~~~~~~~~~~~"
                if c == '0':
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 'n':
                    for n in [300 ,1000]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 'p':
                    for p in [1000, 5000]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 'd':
                    for d in [0.03,0.06]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 'g':
                        for g in [5, 20]:
                            run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 'k':

                        for k in [ 20, 100]:
                            run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 's':
                    for sigX in [5e-6,  2e-5]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c == 'c':
                    for sigY in [0.01, 0.2]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
                if c=='we':
                    for we in [0.01,0.1]:
                        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        # if model=='group':
        #     for seed in range(5):
        #         print "~~~~~~~~~~~~",seed,"~~~~~~~~~~~~~~~~~~~~~~"
        #         if c == '0':
        #             run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 'n':
        #             for n in [300 ,1000]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 'p':
        #             for p in [1000, 5000]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 'gn':
        #             for g_num in [1, 3]: #5
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 'g':
        #             for g in [5, 50]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 'k':
        #             for k in [ 20, 100]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 's':
        #             for sigX in [0.005,0.02]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c == 'c':
        #             for sigY in [0.01, 1]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
        #         if c=='we':
        #             for we in [0.01,0.1]:
        #                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1)
    else:
        print "simulate"
        n = 250
        p = 500
        d = 0.05
        g = 10
        k = 50
        sigX = 1e-5
        sigY=0.1
        seed=0
        we=0.05
        g_num=3
        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, model=model,we=we,g_num=g_num,str1=str1,simulate=simulate)

# def syntheticSingleRun(seed):
#     from utility.syntheticDataGeneration import generateData
#     from sLMMn.sLMMn import run_synthetic
#     roc = True
#     generateData(seed)
#     for j in range(4):
#         run_synthetic(j)
#     for j in range(4):
#         r = evaluateSynthetic(j, roc)
#         print r


if __name__ == '__main__':

    str1='3'
    syntheticRunning(cl='0',str1=str1)
    syntheticRunning(cl='n',str1=str1)
    syntheticRunning(cl='p',str1=str1)


    syntheticRunning(cl='k',str1=str1)
    syntheticRunning(cl='g',str1=str1)

    syntheticRunning(cl='s',str1=str1)
    syntheticRunning(cl='c',str1=str1)
    syntheticRunning(cl='we',str1=str1)

    str1='2'
    syntheticRunning(cl='0',str1=str1)
    syntheticRunning(cl='n',str1=str1)
    syntheticRunning(cl='p',str1=str1)


    syntheticRunning(cl='k',str1=str1)
    syntheticRunning(cl='g',str1=str1)

    syntheticRunning(cl='s',str1=str1)
    syntheticRunning(cl='c',str1=str1)
    syntheticRunning(cl='we',str1=str1)
    syntheticRunning(cl='d',str1=str1)
    str1='3'
    syntheticRunning(cl='d',str1=str1)
    str1='2'
    syntheticRunning(cl='',str1=str1,simulate=True)
    str1='3'
    syntheticRunning(cl='',str1=str1,simulate=True)

