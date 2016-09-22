#from matplotlib import pyplot as plt

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc

from utility import dataLoader


def getPositions(l):
    text = [line.strip() for line in open('causalSnps/athaliana.snps.chromPositionInfo.txt')][1]
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



def gwas_precision_recall(weights, causal_snps, dataType='mice', positions=None, nearby=1000):
    score = np.array(weights)
    label = np.zeros(len(weights))
    if dataType == 'mice':
        for k in causal_snps:
            # label[k] = 1
            # for m in range(k-10, k+11):
            #     if m >=0:
            #         label[m] = 1

            i = np.argmax(score[max(k-nearby,0):k+nearby+1])
            label[i+k-nearby] = 1

        p, r, t = precision_recall_curve(label, score)

        # plt.plot(r[:-1], p[:-1])
        # plt.show()
        return p, r
    elif dataType == 'AT':
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

    # print mp
    # print mi
    return mp, mi

def gwas_nonZero(learnt_weights):
    ind = np.where(learnt_weights!=0)
    return len(ind[0])

def gwas_roc(weights, causal_snps, dataType='mice', positions=None, nearby=1000):
    score = np.array(weights)
    label = np.zeros(len(weights))
    if dataType == 'mice':
        for k in causal_snps:
            # label[k] = 1
            # for m in range(k-10, k+11):
            #     if m >=0:
            #         label[m] = 1

            i = np.argmax(score[max(k-nearby,0):k+nearby+1])
            label[i+k-nearby] = 1

        fpr, tpr, t = roc_curve(label, score)

        # plt.plot(fpr, tpr)
        # plt.show()
        return fpr, tpr
    elif dataType == 'AT':
        if positions is None:
            positions = getPositions(len(score))
        for k in causal_snps:
            mini, maxi = getNearbyIndex(k, positions, nearby)
            i = np.argmax(score[mini:maxi])
            label[mini+i] = 1
        fpr, tpr, t = roc_curve(label, score)

        # plt.plot(fpr, tpr)
        # plt.show()
        return fpr, tpr

def gwas_precision_recall_list(weights_l, causal_snps, legends=None,dataType='mice', positions=None, nearby=1000):
    count = 0
    for weights in weights_l:
        p, r = gwas_precision_recall(weights, causal_snps, dataType, positions, nearby)
        if legends is not None:
            plt.plot(r, p, label=legends[count])
        else:
            plt.plot(r, p)
        count += 1
    plt.ylim(0, 1.05)
    plt.legend()
    plt.show()

def gwas_roc_list(weights_l, causal_snps, legends=None,dataType='mice', positions=None, nearby=1000):
    count = 0
    for weights in weights_l:
        fpr, tpr = gwas_roc(weights, causal_snps, dataType, positions, nearby)
        if legends is not None:
            plt.plot(fpr, tpr, label=legends[count])
        else:
            plt.plot(fpr, tpr)
        count += 1
    plt.legend(loc=4)
    plt.show()

def limitPrediction(l, num):
    s = sorted(l)
    t = s[-num]
    r = []
    for v in l:
        if v > t:
            r.append(v-t)
        else:
            r.append(0)
    return r

def eva(gid, roc, nearby, top, draw=False, fileType='AT', runningMode='dis'):
    scores = []
    if fileType == 'mice':
        snpsFiles = ['10000i','10000n', '10000o','10000s']
        Ks1 = ['', 'square']
        Ks2 = ['', 'Square']
        ls = ['single_K', 'single_K2', 'lasso_K', 'lasso_K2']
        evaNear = 10
    else:
        snpsFiles = ['i', 'n', 's']
        # Ks1 = ['', 'Group', 'square', 'Double']
        # Ks2 = ['', 'Group', 'Square', 'Double']
        # ls = ['single_K','single_G', 'single_K2', 'single_GK', 'lasso_K', 'lasso_G', 'lasso_K2', 'lasso_KG']

        Ks1 = ['', 'square', 'search']
        Ks2 = ['', 'Square', 'Search']
        ls = ['LMM', 'LMM2', 'LMMn', 'sLMM', 'sLMM2', 'sLMMn']
        evaNear = nearby

    fileName = 'result'
    for snpsFile in snpsFiles:
        s = [0,0,0,0,0,0,0,0]
        count = -1
        result = []
        hmaxs = []
        causal = dataLoader.load_data_causal_eva_group(fileType, snpsFile, runningMode, 1)

        for test in ['single', 'lasso']:
            if test == 'single':
                for k in Ks1:
                    count += 1
                    aucs = []
                    sig = np.loadtxt('../multipleRunning/group'+str(gid)+'/dis/Result/'+fileType+'/result_single/'+fileName+k+snpsFile+'.csv', delimiter=',')[:,2]
                    sig = 1-np.array(sig)
                    sig = np.nan_to_num(sig)
                    sigd = limitPrediction(sig, top)
                    if roc:
                        fpr, tpr = gwas_roc(sigd, causal[:,0], fileType, None, evaNear)
                        s[count] = auc(fpr, tpr)
                        if draw:
                            plt.plot(fpr, tpr, label=ls[count]+' AUC:' + str(auc(fpr, tpr)))
                    else:
                        p, r = gwas_precision_recall(sigd, causal[:,0], fileType, None, evaNear)
                        s[count] = auc(r, p)
                        if draw:
                            plt.plot(r, p, label=ls[count])
                count += 1
                aucs = []
                sig = np.loadtxt('../multipleRunning/single/Result_'+str(gid-1)+'/'+fileType+'/result_single/'+fileName+snpsFile+'.csv', delimiter=',')[:,2]
                sig = 1-np.array(sig)
                sig = np.nan_to_num(sig)
                sigd = limitPrediction(sig, top)
                if roc:
                    fpr, tpr = gwas_roc(sigd, causal[:,0], fileType, None, evaNear)
                    s[count] = auc(fpr, tpr)
                    if draw:
                        plt.plot(fpr, tpr, label=ls[count]+' AUC:' + str(auc(fpr, tpr)))
                else:
                    p, r = gwas_precision_recall(sigd, causal[:,0], fileType, None, evaNear)
                    if len(r) >2:
                        s[count] = auc(r, p)
                    else:
                        s[count] = 0
                    if draw:
                        plt.plot(r, p, label=ls[count])
            else:
                for k in Ks2:
                    count += 1
                    bw = np.loadtxt('../multipleRunning/group'+str(gid)+'/'+runningMode+'/Result_'+str(7)+'/'+fileType+'/result_lasso/'+snpsFile+k+'weights.csv', delimiter=',')
                    bw = limitPrediction(bw, top)
                    if roc:
                        fpr, tpr = gwas_roc(bw, causal[:,0], fileType, None, evaNear)
                        s[count] = auc(fpr, tpr)
                        if draw:
                            plt.plot(fpr, tpr, label=ls[count]+' AUC:' + str(auc(fpr, tpr)))
                    else:
                        p, r = gwas_precision_recall(bw, causal[:,0], fileType, None, evaNear)
                        s[count] = auc(r, p)
                        if draw:
                            plt.plot(r, p, label=ls[count])

                count += 1
                bw = np.loadtxt('../multipleRunning/single/Result_'+str(gid-1)+'/'+fileType+'/result_lasso/'+snpsFile+'weights.csv', delimiter=',')
                bw = limitPrediction(bw, top)
                if roc:
                    fpr, tpr = gwas_roc(bw, causal[:,0], fileType, None, evaNear)
                    s[count] = auc(fpr, tpr)
                    if draw:
                        plt.plot(fpr, tpr, label=ls[count]+' AUC:' + str(auc(fpr, tpr)))
                else:
                    p, r = gwas_precision_recall(bw, causal[:,0], fileType, None, evaNear)
                    if len(r) >2:
                        s[count] = auc(r, p)
                    else:
                        s[count] = 0
                    if draw:
                        plt.plot(r, p, label=ls[count])
            if draw:
                plt.ylim(0, 1.05)
                titleSnpsFile = 'Set '
                if snpsFile[-1] == 'i' : titleSnpsFile+='(a)'
                if snpsFile[-1] == 'n' : titleSnpsFile+='(a)'
                # if snpsFile[-1] == 'o' : titleSnpsFile+='3'
                if snpsFile[-1] == 's' : titleSnpsFile+='(b)'
                titleTest = 'single test'
                if test == 'lasso': titleTest = 'combinatorial test'
                if roc:
                    plt.xlabel('false positive rate')
                    plt.ylabel('true positive rate')
                    plt.ylim(0, 0.3)
                    plt.xlim(0, 0.03)
                    if test == 'lasso':
                        plt.legend(loc=4)
                    else:
                        plt.legend(loc=2)
                else:
                    plt.xlabel('recall')
                    plt.ylabel('precision')
                    # plt.xlim(0, 0.4)
                    plt.legend(loc=1)
                plt.title(fileType + ' '+titleSnpsFile + ' '+ titleTest)
                if roc:
                    plt.savefig('figs/roc'+fileType + '_'+ test +'_'+ snpsFile + '.png')
                    plt.clf()
                else:
                    plt.savefig('figs/pr'+fileType + '_'+ test +'_'+ snpsFile + '.png')
                    plt.clf()
        scores.append(s)

    print '--------------------------------'
    print 'start checking:', gid, roc, nearby, top
    print 'check single'
    if scores[0][2] >= max(scores[0][:4])  and scores[1][2] >= max(scores[1][:4]) and scores[2][2] >= max(scores[2][:4]):
        print 'OH perfect!'
    else:
        print 'difference:', scores[0][2] - max(scores[0][:4]), scores[1][2] - max(scores[1][:4]), scores[2][2] - max(scores[2][:4])
    print 'check lasso'
    if scores[0][6] >= max(scores[0][4:])  and scores[1][6] >= max(scores[1][4:]) and scores[2][5] >= max(scores[2][4:]):
        print 'OH perfect!'
    else:
        print 'difference:', scores[0][6] - max(scores[0][4:]), scores[1][6] - max(scores[1][4:]), scores[2][6] - max(scores[2][4:])
    print '--------------------------------'
    print scores




if __name__ == '__main__':

    for gid in [1, 2, 3, 4, 5]:
        for roc in [True, False]:
            for nearby in [1000, 5000, 10000, 50000]:
                for top in [100, 200, 500, 1000]:
                    eva(gid, roc, nearby, top, False)

    # eva(3, True, 50000, 100, True)

    # the most reasonable on multi2 eva(5, True, 10000, 100, True)
    # also reasonable on multi1 eva(3, True, 50000, 100, True)