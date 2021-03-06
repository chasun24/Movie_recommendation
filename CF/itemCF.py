import os
import numpy as np
import random
import math
import time
from numpy import float16


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
        # topMovList_NoSim=self.recmdSys(itemSimRate,trainSetMatrix,friendsNum,movieNum)

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
            ratingsData.append((int(userid),int(itemid),float16(record)-std))     #data列表 [(user,movie,record),(...)...]
        
        print("数据分割为训练集和测试集")
        trainSet=[]
        testSet=[]
        for dataitem in ratingsData:  
            if random.randint(0,99)<testRate:
                testSet.append(dataitem)
            else:
                trainSet.append(dataitem)
        
        print("将训练集和测试集转为矩阵")
        trainSetMatrix = np.full((movieNumMax+1,userNumMax+1), 0, dtype=float16)      #行为movID，列为userID,内容为评分
        for dataitem in trainSet:
            trainSetMatrix[ dataitem[1]][ dataitem[0] ]=dataitem[2]

        testSetMatrix = np.full((movieNumMax+1,userNumMax+1), 0, dtype=float16)  
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
        itemRowInner=np.full((movieNumMax+1, movieNumMax+1), 0, dtype=float16)
        for i in range(1,movieNumMax+1):
            for j in range(i,movieNumMax+1):
                itemRowInner[i][j] =np.dot(trainSetMatrix[i],trainSetMatrix[j])

        #计算item相似度
        itemSimRate = np.full((movieNumMax+1, movieNumMax+1), 0, dtype=float16)
        for i in range(1,movieNumMax+1):     
            for j in range(i+1,movieNumMax+1):
                itemSimRate[i][j]=itemRowInner[i][j] /(math.sqrt(itemRowInner[i][i])*math.sqrt(itemRowInner[j][j]))
                itemSimRate[j][i]=itemSimRate[i][j]
        timend=time.time()

        #保存缓存数据
        np.save(userSimPath,itemSimRate)
        print("计算item相似度itemSimilarity计算时间：{}".format(timend-timest))
        return itemSimRate


#___整个系统用户推荐___
    def recmdSys(self,itemSimRate,trainSetMatrix,friendsNum,movieNum):
        print("开始计算系统推荐...")
        #将uid和similar整理到一个列表用户编号，以1开始
        friendsList=list()  #初始化(userNumMax+1)个列表，且每个列表内为一个空列表
        # uind表示索引，以0开始，uid表示用户
        for uind in range(0,userNumMax):
            friendsList.append(list())
            for find in range(0,userNumMax):
                friendsList[uind].append(list())
                friendsList[uind][find].append(find)
                friendsList[uind][find].append(itemSimRate[uind+1][find+1])
        topFriendList=list()

        # 找到最相似的友邻
        print("找到最相似的{}个友邻".format(friendsNum))
        for uid in range(0,userNumMax):
            topFriendList.append(sorted(friendsList[uid], key=lambda dic: dic[1],reverse=True)[0:friendsNum])

        # 根据相似友邻计算所有电影的推荐指数（全部电影计算，太费时间10min）
        userSimPath="cache/allMovList.npy"
        if self.P_importFromFile and os.path.exists(userSimPath):
            allMovRecedList=np.load(userSimPath)
        else:
            print("根据相似的友邻计算所有电影的推荐指数(about 20 min)")
            timest=time.time()
            allMovRecedList=list()  #use index     [ [ [mid,sum],[mid,sum] ] ,[...]...    ]
            for uind in range(0,userNumMax):
                myUserRecList=list()
                for mid in range(1,movieNumMax+1):
                    sum=0
                    for frd in topFriendList[uind]:
                        fid=frd[0]
                        sim=frd[1]
                        sum+=sim*trainSetMatrix[fid][mid]
                    myUserRecList.append([mid,sum])
                allMovRecedList.append(myUserRecList)
                print("uID:{} 已经计算好!".format(uind+1))
            timend=time.time()
            np.save(userSimPath,allMovRecedList)
            print("推荐用时：{}".format(timend-timest))  

        #按每个用户将推荐列表按推荐指数排序，选择前 movieNum 个
        print("按每个用户将推荐列表按推荐指数排序，选择前 {} 个".format(movieNum))
        topMovRcmdedListPath="cache/topMovRcmdedList.npy"
        if self.P_importFromFile and os.path.exists(topMovRcmdedListPath):
            topMovRcmdedList=np.load(topMovRcmdedListPath)
        else:
            topMovRcmdedList=list()     #use index
            for uind in range(0,userNumMax):
                tamp=allMovRecedList[uind]    #use index
                topMovRcmdedList.append(sorted(tamp,key=lambda x:x[1] ,reverse=True)[0:movieNum])
                print("用户{} ok!".format(uind+1))
            np.save(topMovRcmdedListPath,topMovRcmdedList)

        #去除列表中推荐程度信息，仅有电影编号信息
        print("去除列表中推荐程度信息，仅有电影编号信息")
        topMovList_NoSim=list()
        for uind in range(0,userNumMax):
            myUserList=list()
            tamp=topMovRcmdedList[uind]    #use index
            for itme in tamp:       # [ [mid,sum],[mid,sum].... ]
                myUserList.append(itme[0])
            topMovList_NoSim.append(myUserList)
        return topMovList_NoSim











if __name__=="__main__":
    rate_file=r'./data/small/ratings.csv'
    # fo = open(r'./data/small/result.csv',"w+",encoding="utf-8")
    # fo.write("friendNum,movRecNum,precisionRate,recallRate,absCoverageRate,refCoverageRate,timecost\n")
    itemTest=itemCF(rate_file,10,30)
    
