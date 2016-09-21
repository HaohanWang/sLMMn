import numpy as np
from scipy import linalg

def load_data(fileType, fileName):
    if fileType == 'mice':
        snpsFile = fileName
        Y = np.loadtxt('../expData/snps.'+snpsFile+'.pheno.csv', delimiter=',')
        snps = np.loadtxt('../expData/snps.'+snpsFile[:-1]+'.csv', delimiter=',')
        Kva = np.loadtxt('../expData/snps.'+snpsFile[:-1]+'.Kva.csv', delimiter=',')
        Kve = np.loadtxt('../expData/snps.'+snpsFile[:-1]+'.Kve.csv', delimiter=',')

        causal = np.loadtxt('../expData/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return snps, Y, Kva, Kve, causal
    if fileType == 'AT':
        snpsFile = fileName
        Y = np.loadtxt('../expDataAT/snps.'+snpsFile+'.pheno.csv', delimiter=',')
        snps = np.loadtxt('../data/athaliana.snps.chrom1.csv', delimiter=',')
        Kva = np.loadtxt('../expDataAT/snps.Kva.csv', delimiter=',')
        Kve = np.loadtxt('../expDataAT/snps.Kve.csv', delimiter=',')

        causal = np.loadtxt('../expDataAT/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return snps, Y, Kva, Kve, causal

def load_data_causal(fileType, fileName):
    if fileType == 'mice':
        snpsFile = fileName
        causal = np.loadtxt('../expData/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
    if fileType == 'AT':
        snpsFile = fileName
        causal = np.loadtxt('../expDataAT/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal

def load_data_causal_eva(fileType, fileName, runningMode):
    if fileType == 'mice':
        snpsFile = fileName
        causal = np.loadtxt('../'+runningMode+'/expData/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
    if fileType == 'AT':
        snpsFile = fileName
        causal = np.loadtxt('../'+runningMode+'/expDataAT/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal

def load_data_causal_eva_group(fileType, fileName, runningMode, group):
    if fileType == 'mice':
        snpsFile = fileName
        causal = np.loadtxt('../multipleRunning2/group'+str(group)+'/'+runningMode+'/expData/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
    if fileType == 'AT':
        snpsFile = fileName
        causal = np.loadtxt('../multipleRunning2/group'+str(group)+'/'+runningMode+'/expDataAT/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal

def load_data_at(fileType, fileName, at, runningMode):
    if fileType == 'mice':
        snpsFile = fileName
        if runningMode == 'dis':
            Y = np.loadtxt('../'+runningMode+'/expData/snps.'+snpsFile+'.pheno.csv', delimiter=',')
            causal = np.loadtxt('../'+runningMode+'/expData/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        else:
            Y = np.loadtxt('../'+runningMode+'/expData_'+str(at)+'/snps.'+snpsFile+'.pheno.csv', delimiter=',')
            causal = np.loadtxt('../'+runningMode+'/expData_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        snps = np.loadtxt('../expData/snps.'+snpsFile[:-1]+'.csv', delimiter=',')
        Kva = np.loadtxt('../expData/snps.'+snpsFile[:-1]+'.Kva.csv', delimiter=',')
        Kve = np.loadtxt('../expData/snps.'+snpsFile[:-1]+'.Kve.csv', delimiter=',')

        return snps, Y, Kva, Kve, causal
    if fileType == 'AT':
        snpsFile = fileName
        if runningMode == 'dis':
            Y = np.loadtxt('../'+runningMode+'/expDataAT/snps.'+snpsFile+'.pheno.csv', delimiter=',')
            causal = np.loadtxt('../'+runningMode+'/expDataAT/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        else:
            Y = np.loadtxt('../'+runningMode+'/expDataAT_'+str(at)+'/snps.'+snpsFile+'.pheno.csv', delimiter=',')
            causal = np.loadtxt('../'+runningMode+'/expDataAT_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')

        snps = np.loadtxt('../data/athaliana.snps.chrom1.csv', delimiter=',')
        Kva = np.loadtxt('../expDataAT/snps.Kva.csv', delimiter=',')
        Kve = np.loadtxt('../expDataAT/snps.Kve.csv', delimiter=',')

        return snps, Y, Kva, Kve, causal

def load_data_synthetic(n):
    if n == 3:
        n = 'n'
    else:
        n = str(n)
    snps = np.loadtxt('../syntheticData/X.csv', delimiter=',')
    Y = np.loadtxt('../syntheticData/K'+n+'/y.csv', delimiter=',')
    causal = np.loadtxt('../syntheticData/causal.csv', delimiter=',')
    Kva = np.loadtxt('../syntheticData/Kva.csv', delimiter=',')
    Kve = np.loadtxt('../syntheticData/Kve.csv', delimiter=',')
    return snps, Y, Kva, Kve, causal


def load_data_single_run(runningMode, fileName, at):
    snpsFile = fileName
    Y = np.loadtxt('../'+runningMode+'/expDataAT'+str(at)+'/snps.'+snpsFile+'.pheno.csv', delimiter=',')
    snps = np.loadtxt('../data/athaliana.snps.chrom1.csv', delimiter=',')
    return snps, Y

def load_data_real():
    result = []
    for i in range(1, 6):
        snps = np.loadtxt('../expDataAT2/snps.'+str(i)+'.csv', delimiter=',')
        Kva = np.loadtxt('../expDataAT2/snps.'+str(i)+'.Kva.csv', delimiter=',')
        Kve = np.loadtxt('../expDataAT2/snps.'+str(i)+'.Kve.csv', delimiter=',')
        K = np.loadtxt('../expDataAT2/snps.'+str(i)+'.K.csv', delimiter=',')
        result.append([snps, K, Kva, Kve])
    Y = np.loadtxt('../data/athaliana2.phenos.csv', delimiter=',')
    return result, Y

def load_data_real_at(at):
    Y = np.loadtxt('../data/athaliana2.phenos.csv', delimiter=',')
    Y = Y[:, at]
    result = []
    # print Y
    # print np.where(~np.isnan(Y)==True)[0]
    SNPs = np.loadtxt('../data/athaliana2.snps.csv', delimiter=',')
    SNPs = SNPs[np.where(~np.isnan(Y)==True)[0],:]
    Y = Y[np.where(~np.isnan(Y)==True)[0]]
    text = [line.strip() for line in open('../data/athaliana2.snps.chromPositionInfo.txt')][0]
    chroms = np.array([int(i)for i in text.split('\t')])
    for i in range(1,6):
        snps1 = SNPs[:, np.where(chroms==i)[0]]
        snps2 = SNPs[:, np.where(chroms!=i)[0]]
        K = np.dot(snps2, snps2.T)
        Kva, Kve = linalg.eigh(K)
        result.append([snps1, K, Kva, Kve])

    return result, Y

def load_group_info(fileType):
    if fileType == 'mice':
        pass
    else:
        group = np.loadtxt('../data/athaliana.snps.categories.txt')
        maxi = np.max(group)
        nd = np.zeros([group.shape[0], maxi+1])
        for i in range(group.shape[0]):
            nd[i, group[i]] = 1
        # group = group.reshape([group.shape[0],1])
        return nd

def load_data_causal_at(fileType, fileName, at):
    if fileType == 'mice':
        snpsFile = fileName
        causal = np.loadtxt('../expData_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
    if fileType == 'AT':
        snpsFile = fileName
        causal = np.loadtxt('../expDataAT_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal

def load_data_causal_at_eva(fileType, fileName, at, runningMode):
    if fileType == 'mice':
        snpsFile = fileName
        causal = np.loadtxt('../'+runningMode+'/expData_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
    if fileType == 'AT':
        snpsFile = fileName
        causal = np.loadtxt('../'+runningMode+'/expDataAT_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal

def load_data_causal_at_eva_group(fileType, fileName, at, runningMode, group):
    if fileType == 'mice':
        snpsFile = fileName
        causal = np.loadtxt('../multipleRunning/group'+str(group)+'/'+runningMode+'/expData_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
    if fileType == 'AT':
        snpsFile = fileName
        causal = np.loadtxt('../multipleRunning/group'+str(group)+'/'+runningMode+'/expDataAT_'+str(at)+'/snps.'+snpsFile+'.pheno.causal.csv', delimiter=',')
        return causal
