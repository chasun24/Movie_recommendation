import os
import numpy as np
import random
import math
import time



class itemCF:

    global std,userNumMax,movieNumMax,testRate
    std=3.0     #定义常变量表示标准，小于3表示不喜爱，大于3表示喜爱
    userNumMax=610
    movieNumMax=193609
    testRate=10     #测试集占源数据集的比率%
    P_importFromFile = False #是否可以从文件读入

    precisionRate=0
    recallRate=0
    absCoverageRate=0
    refCoverageRate=0


#___开始___
    def __init__(self,rate_file,friendsNum,movieNum):

        trainSetMatrix,testSetMatrix=self.dataLoadAndSplit(rate_file)
        itemSimRate=self.itemSimFunc(trainSetMatrix)
        pass


#___数据返回____
    def dataReturn(self):
        return self.precisionRate,self.recallRate,self.absCoverageRate,self.refCoverageRate

#___数据载入___
    def dataLoadAndSplit(self,rate_file):
        print("从文件读入数据")
        ratingsData=[]
        for line in open(rate_file):
            userid,itemid,record,idcode = line.split(",")
            ratingsData.append((int(userid),int(itemid),float(record)-std))     #data列表 [(user,movie,record),(...)...]
        
        print("数据分割为训练集和测试集")
        trainSet=[]
        testSet=[]
        for dataitem in ratingsData:  
            if random.randint(0,99)<testRate:
                testSet.append(dataitem)
            else:
                trainSet.append(dataitem)
        
        print("将训练集和测试集转为矩阵")
        trainSetMatrix = np.full((movieNumMax+1,userNumMax+1), 0, dtype=float)      #行为movID，列为userID,内容为评分
        for dataitem in trainSet:
            trainSetMatrix[ dataitem[1]][ dataitem[0] ]=dataitem[2]

        testSetMatrix = np.full((movieNumMax+1,userNumMax+1), 0, dtype=float)  
        for dataitem in testSet:
            testSetMatrix[ dataitem[1]][ dataitem[0] ]=dataitem[2]
        return  trainSetMatrix,testSetMatrix


#___用户相似度矩阵___
    def itemSimFunc(self,trainSetMatrix):
        userSimPath="cache/itemCF_item_Similarity.npy"
        if self.P_importFromFile and os.path.exists(userSimPath):
            print("从文件导入item相似度 ...")
            return np.load(userSimPath)
        print("开始计算item相似度 ...")
        timest=time.time()

        #计算两个movId向量的内积(包括了自身)
        itemRowInner=np.full((movieNumMax+1, movieNumMax+1), 0, dtype=float)
        for i in range(1,movieNumMax+1):
            for j in range(i,movieNumMax+1):
                itemRowInner[i][j] =np.dot(trainSetMatrix[i],trainSetMatrix[j])

        #计算item相似度
        itemSimRate = np.full((userNumMax+1, userNumMax+1), 0, dtype=float)
        for i in range(1,userNumMax+1):     
            for j in range(i+1,userNumMax+1):
                itemSimRate[i][j]=itemRowInner[i][j] /(math.sqrt(itemRowInner[i][i])*math.sqrt(itemRowInner[j][j]))
                itemSimRate[j][i]=itemSimRate[i][j]
        timend=time.time()
        np.save(userSimPath,itemSimRate)
        print("计算item相似度itemSimilarity计算时间：{}".format(timend-timest))
        return itemSimRate












if __name__=="__main__":
    rate_file=r'./data/small/ratings.csv'
    # fo = open(r'./data/small/result.csv',"w+",encoding="utf-8")
    # fo.write("friendNum,movRecNum,precisionRate,recallRate,absCoverageRate,refCoverageRate,timecost\n")
    friendNumChoice=[30,40,50]
    movRecNumChoice=[10,20,30,40,50]
    for i in range(0,len(friendNumChoice)):
        for j in range(0,len(movRecNumChoice)):
            fo = open(r'./data/small/result.csv',"a+",encoding="utf-8")
            tst=time.time()
            cfTest=userCF(rate_file,friendNumChoice[i],movRecNumChoice[j])
            ls=cfTest.dataReturn()
            tend=time.time()
            tcost=tend-tst  
            fo.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(friendNumChoice[i],movRecNumChoice[j],ls[0],ls[1],ls[2],ls[3],tcost))
            fo.close()
    
