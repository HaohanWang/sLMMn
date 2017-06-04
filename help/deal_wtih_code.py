import numpy as np

def loadResult2(str1,n, p, d, g, sig, sigC,k,g_num,we, seed):
    pathTail2 =  str1+'_'+str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(sig) + '_' + str(sigC) + '_' + str(seed) + '_'

    pathTail =  str1+'_'+str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(sig) + '_' + str(sigC) + '_' +str(we)+'_'+str(g_num)+ '_'+str(seed) + '_'
    results  =  np.load('../result/synthetic/group2/' + pathTail2+'beta1.npy')
    results4  =  np.load('../result/synthetic/group2/' + pathTail2+'beta2.npy')
    results2  =  np.load('../result/synthetic/group2/' + pathTail2+'X.npy')
    results3 =  np.load('../result/synthetic/group2/' + pathTail2+'Y.npy')
    print pathTail2
    print pathTail

    print results.shape
    print results2.shape
    print results3.shape
    print results4.shape
    return results,results2,results3,results4,pathTail


def ReplaceResult(cat,str1=''):
    n = 100
    p = 500
    d = 0.05
    g = 10
    k = 50
    sig = 0.001
    sigC=0.1
    we=0.05
    g_num = 3
    if cat == 'n':
        valueList = [ 50,100,500]
    elif cat == 'p':
        valueList = [200, 800]
    elif cat=='k':
        valueList = [20,100]
    # elif cat == 'd':
    #     valueList = [0.01]
    elif cat == 'g':
        valueList =[ 5,20]
    elif cat=='gn':
        valueList= [2,5]
    elif cat == 's':
        valueList = [0.01]
    elif cat=='c':
        valueList = [1]
    elif cat=='we':
        valueList =[0.01 ,0.1]
    print valueList
    for j in valueList:
        if cat == 'n':
            n=j
        elif cat == 'p':
            p=j
        elif cat=='k':
            k=j
        elif cat == 'd':
            d=j
        elif cat == 'g':
            g=j
        elif cat=='gn':
            g_num=j
        elif cat == 's':
            sig=j
        elif cat=='c':
            sigC=j
        elif cat=='we':
            we=j
        for i in range(5):
            r1,r2,r3,r4,pathTail=loadResult2(str1,n, p, d, g, sig, sigC,k,g_num,we, i)

            np.save('../result/synthetic/group/' + pathTail+'beta1', r1)
            np.save('../result/synthetic/group/' + pathTail+'X', r2)
            np.save('../result/synthetic/group/' + pathTail+'Y', r3)
            np.save('../result/synthetic/group/' + pathTail+'beta2', r4)



if __name__ == '__main__':
    for str1 in ['2','3','n']:
        for cat in ['n','p','k','g','c','s']: #,'we' ,'g'
            print "--------------",cat

            ReplaceResult(cat,str1)




