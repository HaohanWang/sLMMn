__author__ = 'Haohan Wang'

from evaluation import *
from sklearn.metrics import auc
import matplotlib.ticker as ticker

def evaluate(runningMode, fileType, roc):

    # runningMode = 'ps' # 'dis', 'ph', 'pe', 'ps', 'snp'
    #
    #
    # fileType = 'mice'

    snpsFiles = ['s']
    Ks1 = ['', 'square', 'search']
    Ks2 = ['',  'Square', 'Search']
    ls1 = ['LMM', 'LMM2', 'LMMn']
    ls2 = ['sLMM', 'sLMM2', 'sLMMn']
    evaNear = 10000

    fileName = 'result'
    for snpsFile in snpsFiles:
        for test in ['single', 'lasso']:
            result = []
            error = []
            resultAll = []
            hmaxs = []
            if test == 'single':
                for k in Ks1:
                    if runningMode == 'dis':
                        aucs = []
                        for groupID in range(1, 5):
                            aucs.append([])
                            causal = dataLoader.load_data_causal_eva_group(fileType, snpsFile, runningMode, groupID)
                            # discovers = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
                            discovers = [20, 50, 100, 500, 2000]
                            sig = np.loadtxt('../multipleRunning5/group'+str(groupID)+'/dis/Result/'+fileType+'/result_single/'+fileName+k+snpsFile+'.csv', delimiter=',')[:,2]
                            sig = 1-np.array(sig)
                            sig = np.nan_to_num(sig)
                            for d in discovers:
                                sigd = limitPrediction(sig, d)
                                if roc:
                                    fpr, tpr = gwas_roc(sigd, causal[:,0], fileType, None, evaNear)
                                    a = auc(fpr, tpr)
                                else:
                                    p, r = gwas_precision_recall(sigd, causal[:,0], fileType, None, evaNear)
                                    a = auc(r, p)
                                aucs[groupID-1].append(a)
                        aucs_ = []
                        errs_ = []
                        for i in range(5):
                            aucs_.append(np.mean([k[i] for k in aucs]))
                            errs_.append(np.std([k[i] for k in aucs])/5)
                        result.append(aucs_)
                        error.append(errs_)
                        resultAll.append(aucs)
                        # hs = [float(line.strip()) for line in open('../multipleRunning2/group'+str(groupID)+'/dis/Result/'+fileType+'/result_single/'+fileName+k+snpsFile+'.hmax.csv')]*10
                        # hmaxs.append(hs)
                    elif runningMode == 'snp':
                        aucs = []
                        # hs = []
                        for groupID in range(1, 5):
                            aucs.append([])
                            for fileSet in [0, 2, 3, 4, 8]:
                                causal = dataLoader.load_data_causal_at_eva_group(fileType, snpsFile, fileSet, runningMode, groupID)
                                sig = np.loadtxt('../multipleRunning5/group'+str(groupID)+'/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_single/'+fileName+k+snpsFile+'.csv', delimiter=',')[:,2]
                                sig = 1-np.array(sig)
                                sig = np.nan_to_num(sig)
                                sigd = limitPrediction(sig, 500)
                                if roc:
                                    fpr, tpr = gwas_roc(sigd, causal[:,0], fileType, None, evaNear)
                                    a = auc(fpr, tpr)
                                else:
                                    p, r = gwas_precision_recall(sigd, causal[:,0], fileType, None, evaNear)
                                    a = auc(r, p)
                                aucs[groupID-1].append(a)
                            # h = [float(line.strip()) for line in open('../multipleRunning2/group1/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_single/'+fileName+k+snpsFile+'.hmax.csv')][0]
                            # hs.append(h)
                        aucs_ = []
                        errs_ = []
                        for i in range(5):
                            aucs_.append(np.mean([k[i] for k in aucs]))
                            errs_.append(np.std([k[i] for k in aucs])/5)
                        result.append(aucs_)
                        error.append(errs_)
                        resultAll.append(aucs)

                    else:
                        aucs = []
                        # hs = []
                        for groupID in range(1, 5):
                            aucs.append([])
                            for fileSet in range(1, 10, 2):
                                causal = dataLoader.load_data_causal_at_eva_group(fileType, snpsFile, fileSet, runningMode, groupID)
                                sig = np.loadtxt('../multipleRunning5/group'+str(groupID)+'/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_single/'+fileName+k+snpsFile+'.csv', delimiter=',')[:,2]
                                sig = 1-np.array(sig)
                                sig = np.nan_to_num(sig)
                                sigd = limitPrediction(sig, 500)
                                if roc:
                                    fpr, tpr = gwas_roc(sigd, causal[:,0], fileType, None, evaNear)
                                    a = auc(fpr, tpr)
                                else:
                                    p, r = gwas_precision_recall(sigd, causal[:,0], fileType, None, evaNear)
                                    a = auc(r, p)
                                aucs[groupID-1].append(a)
                            # h = [float(line.strip()) for line in open('../multipleRunning2/group1/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_single/'+fileName+k+snpsFile+'.hmax.csv')][0]
                            # hs.append(h)
                        aucs_ = []
                        errs_ = []
                        for i in range(5):
                            aucs_.append(np.mean([k[i] for k in aucs]))
                            errs_.append(np.std([k[i] for k in aucs])/5)
                        result.append(aucs_)
                        error.append(errs_)
                        resultAll.append(aucs)
                        # hmaxs.append(hs)
                x = (np.arange(5) + 1).astype(int)
                for i in range(len(result)):
                    # plt.plot(x, result[i], label=ls[i])
                    plt.errorbar(x, result[i], yerr=error[i], label=ls1[i]+' average: '+str(round(np.mean(result[i]), 4)))
                plt.legend(loc=1)
                plt.xlabel('index of parameter')
                plt.ylabel('average area under ROC')
                plt.xlim(0, 6)
                # plt.axis([1, 2, 3, 4, 5])
                if roc:
                    plt.savefig('../figs/single_ROC_'+fileType + '_'+ runningMode +'_'+ snpsFile + '.png')
                else:
                    plt.savefig('../figs/single_PR_'+fileType + '_'+ runningMode +'_'+ snpsFile + '.png')
                plt.clf()
            else:
                result = []
                error = []
                resultAll = []
                hmaxs = []
                for k in Ks2:
                    aucs = []
                    if runningMode == 'dis':
                        for groupID in range(1, 5):
                            aucs.append([])
                            for fileSet in [2, 3, 4, 6, 8]:
                                if runningMode != 'dis':
                                    causal = dataLoader.load_data_causal_at_eva_group(fileType, snpsFile, fileSet, runningMode, groupID)
                                else:
                                    causal = dataLoader.load_data_causal_eva_group(fileType, snpsFile, runningMode, groupID)
                                bw = np.loadtxt('../multipleRunning5/group'+str(groupID)+'/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_lasso/'+snpsFile+k+'weights.csv', delimiter=',')
                                bw = limitPrediction(bw, 500)
                                if roc:
                                    fpr, tpr = gwas_roc(bw, causal[:,0], fileType, None, evaNear)
                                    a = auc(fpr, tpr)
                                else:
                                    p, r = gwas_precision_recall(bw, causal[:,0], fileType, None, evaNear)
                                    a = auc(r, p)
                                aucs[groupID-1].append(a)
                    elif runningMode == 'snp':
                        for groupID in range(1, 5):
                            aucs.append([])
                            for fileSet in [0, 2, 3, 4, 8]:
                                if runningMode != 'dis':
                                    causal = dataLoader.load_data_causal_at_eva_group(fileType, snpsFile, fileSet, runningMode, groupID)
                                else:
                                    causal = dataLoader.load_data_causal_eva_group(fileType, snpsFile, runningMode, groupID)
                                bw = np.loadtxt('../multipleRunning5/group'+str(groupID)+'/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_lasso/'+snpsFile+k+'weights.csv', delimiter=',')
                                bw = limitPrediction(bw, 500)
                                if roc:
                                    fpr, tpr = gwas_roc(bw, causal[:,0], fileType, None, evaNear)
                                    a = auc(fpr, tpr)
                                else:
                                    p, r = gwas_precision_recall(bw, causal[:,0], fileType, None, evaNear)
                                    a = auc(r, p)
                                aucs[groupID-1].append(a)
                    else:
                        for groupID in range(1, 5):
                            aucs.append([])
                            for fileSet in range(1,10,2):
                                if runningMode != 'dis':
                                    causal = dataLoader.load_data_causal_at_eva_group(fileType, snpsFile, fileSet, runningMode, groupID)
                                else:
                                    causal = dataLoader.load_data_causal_eva_group(fileType, snpsFile, runningMode, groupID)
                                bw = np.loadtxt('../multipleRunning5/group'+str(groupID)+'/'+runningMode+'/Result_'+str(fileSet)+'/'+fileType+'/result_lasso/'+snpsFile+k+'weights.csv', delimiter=',')
                                bw = limitPrediction(bw, 500)
                                if roc:
                                    fpr, tpr = gwas_roc(bw, causal[:,0], fileType, None, evaNear)
                                    a = auc(fpr, tpr)
                                else:
                                    p, r = gwas_precision_recall(bw, causal[:,0], fileType, None, evaNear)
                                    a = auc(r, p)
                                aucs[groupID-1].append(a)
                    aucs_ = []
                    errs_ = []
                    for i in range(5):
                        aucs_.append(np.mean([k[i] for k in aucs]))
                        errs_.append(np.std([k[i] for k in aucs])/5)
                    result.append(aucs_)
                    error.append(errs_)
                    resultAll.append(aucs)
                x = (np.arange(5) + 1).astype(int)
                for i in range(len(result)):
                    # plt.plot(x, result[i], label=ls[i])
                    plt.errorbar(x, result[i], yerr=error[i], label=ls2[i]+' average: '+str(round(np.mean(result[i]), 4)))
                plt.legend(loc=1)
                plt.xlabel('index of parameter')
                plt.ylabel('average area under ROC')
                plt.xlim(0, 6)
                # plt.axis([1, 2, 3, 4, 5])
                if roc:
                    plt.savefig('../figs/lasso_ROC_'+fileType + '_'+ runningMode +'_'+ snpsFile + '.png')
                else:
                    plt.savefig('../figs/lasso_PR_'+fileType + '_'+ runningMode +'_'+ snpsFile + '.png')
                plt.clf()

        # for gID in range(0, 5):
        #     for t in range(len(resultAll)):
        #         # print gID, t, len(resultAll), len(resultAll[t])
        #         plt.plot(x, resultAll[t][gID], label=ls[t])
        #     plt.legend(loc=1)
        #     plt.title(fileType + runningMode + snpsFile)
        #     if roc:
        #         plt.savefig('../figs/group'+str(gID+1)+'/ROC_'+fileType + '_'+ runningMode +'_'+ snpsFile + '.png')
        #     else:
        #         plt.savefig('../figs/group'+str(gID+1)+'/PR_'+fileType + '_'+ runningMode +'_'+ snpsFile + '.png')
        #     plt.clf()

        # plt.show()

        # calculate area (the following)
        # print fileType, runningMode, snpsFile,
        # m = np.mean(result, axis=1)
        # # for k in m:1
        # #     print k, '\t',
        # # print
        # sym1 = ''
        # sym2 = ''
        # sym3 = ''
        # sym4 = ''
        # if m[0] > 1.01*m[1]: sym1+='*'
        # if m[0] > 1.05*m[1]: sym1+='#'
        # if m[1] > 1.01*m[0]: sym2+='*'
        # if m[1] > 1.05*m[0]: sym2+='#'
        # if m[2] > 1.01*m[3]: sym3+='*'
        # if m[2] > 1.05*m[3]: sym3+='#'
        # if m[3] > 1.01*m[2]: sym4+='*'
        # if m[3] > 1.05*m[2]: sym4+='#'
        # print m[0],sym1+'\t',
        # print m[1],sym2+'\t',
        # print m[2],sym3+'\t',
        # print m[3],sym4
        # calculate area (the above)

        # for i in range(2):
        #     print hmaxs[i]
        #     print result[i]

if __name__ == '__main__':
    print 'WARNING: This script is slow'
    fileType = 'AT'
    for roc in [True]:
        for runningMode in ['snp', 'ph', 'ps', 'pe', 'dis']:
            evaluate(runningMode, fileType, roc)