import os
import numpy as np
import random
import math
import time
import heapq


class userCF:
    global std,userNumMax,movieNumMax,testRate
    std=3.0     #定义常变量表示标准，小于3表示不喜爱，大于3表示喜爱
    userNumMax=610
    movieNumMax=193609
    testRate=10     #测试集占源数据集的比率
#___开始___
    def __init__(self,rate_file,friendsNum,movieNum):

        
        trainSetMatrix,testSetMatrix=self.dataLoadAndSplit(rate_file)
        userSimilarity=self.userSimFunc(trainSetMatrix)
        # userId=1        #test
        topMovList_NoSim=self.recmdSys(userSimilarity,trainSetMatrix,friendsNum,movieNum)
        self.precisionAndRecall(topMovList_NoSim,trainSetMatrix,movieNum)

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
        trainSetMatrix = np.full((userNumMax+1, movieNumMax+1), 0, dtype=float)  
        for dataitem in trainSet:
            trainSetMatrix[ dataitem[0]][ dataitem[1] ]=dataitem[2]
        testSetMatrix = np.full((userNumMax+1, movieNumMax+1), 0, dtype=float)  
        for dataitem in testSet:
            testSetMatrix[ dataitem[0]][ dataitem[1] ]=dataitem[2]
        return  trainSetMatrix,testSetMatrix

#___用户相似度矩阵___
    def userSimFunc(self,trainSetMatrix):
        userSimPath="cache/userSimilarity.npy"
        if os.path.exists(userSimPath):
            print("从文件导入用户相似度 ...")
            return np.load(userSimPath)
        print("开始计算用户相似度 ...(about 30 s)")
        timest=time.time()
        #计算两个向量的内积(包括了自身)
        userRowInner=np.full((userNumMax+1, userNumMax+1), 0, dtype=float)
        for i in range(1,userNumMax+1):
            for j in range(i,userNumMax+1):
                userRowInner[i][j] =np.dot(trainSetMatrix[i],trainSetMatrix[j])
        #计算用户相似度
        userSimilarity = np.full((userNumMax+1, userNumMax+1), 0, dtype=float)
        for i in range(1,userNumMax+1):     
            for j in range(i+1,userNumMax+1):
                userSimilarity[i][j]=userRowInner[i][j] /(math.sqrt(userRowInner[i][i])*math.sqrt(userRowInner[j][j]))
                userSimilarity[j][i]=userSimilarity[i][j]
        timend=time.time()
        np.save(userSimPath,userSimilarity)
        print("计算用户相似度userSimilarity计算时间：{}".format(timend-timest))
        return userSimilarity

#___整个系统用户推荐___
    def recmdSys(self,userSimilarity,trainSetMatrix,friendsNum,movieNum):
        print("开始计算系统推荐...")
        #将uid和similar整理到一个列表用户编号，以1开始
        friendsList=list()  #初始化(userNumMax+1)个列表，且每个列表内为一个空列表
        # uind表示索引，以0开始，uid表示y
        for uind in range(0,userNumMax):
            friendsList.append(list())
            for find in range(0,userNumMax):
                friendsList[uind].append(list())
                friendsList[uind][find].append(find)
                friendsList[uind][find].append(userSimilarity[uind+1][find+1])
        topFriendList=list()
        # 找到最相似的友邻
        for uid in range(0,userNumMax):
            topFriendList.append(sorted(friendsList[uid], key=lambda dic: dic[1],reverse=True)[0:friendsNum])
        # 根据相似友邻计算所有电影的推荐指数（全部电影计算，太费时间10min）
        userSimPath="cache/allMovList.npy"
        if os.path.exists(userSimPath):
            allMovRecedList=np.load(userSimPath)
        else:
            print("根据相似友邻计算所有电影的推荐指数(about 20 min)")
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
        if os.path.exists(topMovRcmdedListPath):
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


#___系统准确率和召回率___
    def precisionAndRecall(self,topMovList_NoSim,trainSetMatrix,movieNum):
        '''
        准确率=每个用户的(预测电影列表和测试集电影列表交集的数量)之和allUserSameMovNum/每个用户的(测试集电影列表数量)之和allUserTestMovNum
        召回率=每个用户的(预测电影列表和测试集电影列表交集的数量)之和allUserSameMovNum/每个用户的(预测电影列表数量)之和allUserPdicMovNum
        '''
        print("计算系统准确率和召回率(about 90s)")
        timest=time.time()
        allUserSameMovNum=0
        allUserTestMovNum=0
        for uid in range(1,userNumMax+1):
            uind=uid-1
            sameMovNum=0
            testMovNum=0
            testList=trainSetMatrix[uid]
            predictList=topMovList_NoSim[uind]   #use index
            for Tmid in range(1,movieNumMax+1):
                rec=testList[Tmid]
                if rec<=0:
                    continue
                testMovNum+=1;
                for Pm in predictList:
                    if Tmid==Pm:
                        sameMovNum+=1
            # print("==>  {}/{}".format(uind+1,userNumMax))
            allUserSameMovNum+=sameMovNum
            allUserTestMovNum+=testMovNum
        allUserPdicMovNum=userNumMax*movieNum
        precision=allUserSameMovNum/allUserTestMovNum
        reCall=allUserSameMovNum/allUserPdicMovNum
        timend=time.time()
        print("计算准确率和召回率的时间：{}".format(timend-timest))
        print("<===================================>")
        print("准确率：{:.4f}%".format(precision*100))
        print("召回率：{:.4f}%".format(reCall*100))
if __name__=="__main__":
    rate_file=r'./data/small/ratings.csv'
    userCF=userCF(rate_file,10,40)




