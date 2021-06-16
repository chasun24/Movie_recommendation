import os
import numpy as np
import random
import math
import time
import gc


class userCF:
    global std,userNumMax,movieNumMax,testRate
    std=3.0     #定义常变量表示标准，小于3表示不喜爱，大于3表示喜爱
    userNumMax=610
    movieNumMax=193609
    testRate=10     #测试集占源数据集的比率%
    P_importFromFile = True #是否可以从文件读入

    precisionRate=0
    recallRate=0
    absCoverageRate=0
    refCoverageRate=0

#___开始___
    def __init__(self,rate_file,friendsNum,movieNum):

        
        trainSetMatrix,activeUsrList,testSetList,userWatched=self.dataLoadAndSplit(rate_file)
        userSimilarity=self.userSimFunc(trainSetMatrix)
        topMovList_NoSim=self.recmdSys(userSimilarity,trainSetMatrix,friendsNum,movieNum,userWatched)
        self.precisionAndRecall(topMovList_NoSim,testSetList)
        self.coverage(topMovList_NoSim)


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

        print("计算活跃用户列表，即评论数>500的用户")
        activeUsrList=[]    # record uIdx whose ratingNums > 500
        u_ratingNum = np.full( userNumMax, 0, dtype=int)  # index:uIdx value:ratingNums
        for dataitem in ratingsData:
            uIdx=dataitem[0]-1
            u_ratingNum[uIdx] +=1
        for uIdx in range(0,userNumMax):
            if u_ratingNum[uIdx]>500:
                activeUsrList.append(uIdx)


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

        testSetList = []
        for i  in range(0,userNumMax):
            testSetList.append(list())
        for dataitem in testSet:
            uIdx=dataitem[0]-1
            mId=dataitem[1]
            testSetList[uIdx].append(mId)
        
        userWatched=[]
        for i in range(0,userNumMax):
            userWatched.append(list())
        for item in ratingsData:
            uIdx=dataitem[0]-1
            userWatched[uIdx].append(item[1])
            
        return  trainSetMatrix,activeUsrList,testSetList,userWatched

#___用户相似度矩阵___
    def userSimFunc(self,trainSetMatrix):
        userSimPath="cache/userSimilarity.npy"
        if self.P_importFromFile and os.path.exists(userSimPath):
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
        del userRowInner
        gc.collect()
        return userSimilarity

#___整个系统用户推荐___
    def recmdSys(self,userSimilarity,trainSetMatrix,friendsNum,movieNum,userWatched):
        
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
        del userSimilarity
        gc.collect()
        # 找到最相似的友邻
        print("找到最相似的{}个友邻".format(friendsNum))
        for uid in range(0,userNumMax):
            topFriendList.append(sorted(friendsList[uid], key=lambda dic: dic[1],reverse=True)[0:friendsNum])
        # 根据相似友邻计算所有电影的推荐指数（全部电影计算，太费时间10min）
        del friendsList
        gc.collect()
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
                u_topList=sorted(tamp,key=lambda x:x[1] ,reverse=True)
                u_recList=[]
                cnt=0
                for item in u_topList:
                    mId=int(item[0])
                    if mId not in userWatched[uind]:
                        print(mId)
                        u_recList.append(mId)
                        cnt+=1
                        if cnt>= movieNum:
                            break

                topMovRcmdedList.append(u_recList)
                print("用户{} ok!".format(uind+1))
                if uind==608:
                    print("608")
            # print("保存topMovRcmdedList.npy")
            # np.save(topMovRcmdedListPath,topMovRcmdedList)
        print("回收 allMovRecedList")
        del allMovRecedList
        gc.collect()

        return topMovRcmdedList


#___系统准确率和召回率___
    def precisionAndRecall(self,topMovList_NoSim,testSetList):
        '''
        准确率=每个用户的(预测电影列表和测试集电影列表交集的数量)之和allUserSameMovNum/每个用户的(测试集电影列表数量)之和 allUserTestMovNum
        召回率=每个用户的(预测电影列表和测试集电影列表交集的数量)之和allUserSameMovNum/每个用户的(预测电影列表数量)之和 allUserPdicMovNum
        '''
        print("计算非活跃用户和活跃用户的的准确率和召回率")

        Act_allUserSameMovNum=0
        Act_allUserTestMovNum=0
        Act_allUserPdicMovNum=0
        for uIdx in range(0,userNumMax):
            u_topMList=topMovList_NoSim[uIdx]
            u_testMList=testSetList[uIdx]
            IntersectionNum= len(list(set(u_topMList) & set(u_testMList)))
            Act_allUserSameMovNum+=IntersectionNum
            Act_allUserTestMovNum+=len(u_testMList)
            Act_allUserPdicMovNum+=len(u_topMList)


        Act_precision=Act_allUserSameMovNum/Act_allUserTestMovNum
        Act_reCall=Act_allUserSameMovNum/Act_allUserPdicMovNum
        
        print("<============ACTIVE_USER===========>")
        print("准确率：{:.4f}%".format(Act_precision*100))
        print("召回率：{:.4f}%".format(Act_reCall*100))

        
#___系统覆盖率___
    def coverage(self,topMovList_NoSim):
        '''
        覆盖率=每个用户的预测电影列表的集合/系统电影总数量  (至多为 userNumMax*movieNum/movieNumMax ，此时为每个用户推荐的电影都不同)
        在覆盖率不可能达到100%的情况下定义：
            1. 绝对覆盖率
                每个用户的预测电影列表的交集/系统电影总数量 （源定义）
            2. 相对覆盖率
                每个用户的预测电影列表的交集/系统为用户推荐的总电影数   (至多可达100%)
        '''
        print("计算活跃用户和非活跃用户的覆盖率")
        Act_movSet=set()
        Act_allRecNum=0
        for uIdx in range(0,userNumMax):
            u_topMSet=set(topMovList_NoSim[uIdx])
            Act_movSet=Act_movSet | u_topMSet
            Act_allRecNum+=len(u_topMSet)
        Act_SetNum=len(Act_movSet)

        print("<===========ACT_USER=============>")
        print("相对覆盖率：{:.6f}%".format(Act_SetNum/Act_allRecNum*100))
if __name__=="__main__":
    rate_file=r'./data/small/ratings.csv'
    tset=userCF(rate_file,10,30)