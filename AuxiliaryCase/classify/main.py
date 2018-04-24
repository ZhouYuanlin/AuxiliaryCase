#coding:utf-8

from lntegrated_algorithm import *
from multiprocessing import Pool

def processKeyPeopleClassfic():
    p = Pool(len(AlgorithmDirection.all_classify.value))
    for type in AlgorithmDirection.all_classify.value.keys():
        p.apply_async(keyPeopleClassfic,args=(type,))
    p.close()
    p.join()
    optimization_result(groupOrPeople=GroupOrPeople.QQInX)

def processKeyGroupClassfic():
    KeyWordClassfic()
    optimization_result(groupOrPeople=GroupOrPeople.GroupInX)



if __name__ == "__main__":
    s = time.time()
    processKeyGroupClassfic()
    print "%ds"%(time.time()-s)
