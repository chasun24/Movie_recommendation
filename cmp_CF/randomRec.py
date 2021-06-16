import os
import numpy as np
import random
import math
import time
import pymysql


class itemCF:
    global std,userNumMax,movieNumMax,NEIGHBOR_NUM,REC_MOV_NUM,testRate
    std=3.0     #定义常变量表示标准，小于3表示不喜爱，大于3表示喜爱
    userNumMax=0
    movieNumMax=0
    movieIdMax=193609
    NEIGHBOR_NUM=10
    REC_MOV_NUM=30
    testRate=10
    


#___开始___
    def __init__(self,rate_file,friendsNum=10,movieNum=30):
        print("=========开始使用基于物品推荐===========")
        START=time.time()
        movieFun,trainSetList,testSetList=self.setUser_MovieNum_MoiveFun(rate_file)
        recMovList=self.recommend(movieFun,trainSetList)
        self.precisionAndRecall(recMovList,testSetList)
        self.coverage(recMovList)
        
        END=time.time()
        COST=END-START
        print("用时：",COST)


#___设置userNumMax, movieNumMax___
    def setUser_MovieNum_MoiveFun(self,rate_file):
        print("从文件读入数据")
        ratingList=[]
        movieSet=set()
        userSet=set()
        for line in open(rate_file):
            userid,itemid,record,idcode = line.split(",")
            ratingList.append((int(userid),int(itemid),float(record)))     #data列表 [(user,movie,record),(...)...]
            userSet.add(int(userid))
            movieSet.add(int(itemid))
        movieFun=sorted(list(movieSet))
        print("")
        # 从此 movieId 可以从 movieFun中取 movieId=movieFun[movieIdx] <===> moiveIdx=movieFun.index(movieId) 
        # 即 可以用 mIdx 代替 mId 计算
        global userNumMax, movieNumMax
        userNumMax = len(userSet)  
        movieNumMax= len(movieSet) 
        print("电影movieNumMax》》》》》》",movieNumMax)

        print("数据分割为训练集和测试集")
        trainSet=[]
        testSet=[]
        for dataitem in ratingList:  
            if random.randint(0,99)<testRate:
                testSet.append(dataitem)
            else:
                trainSet.append(dataitem)

        trainSetList=[]
        for i  in range(0,userNumMax):
            trainSetList.append(list())
        for dataitem in trainSet:
            uIdx=dataitem[0]-1
            mId=dataitem[1]
            trainSetList[uIdx].append(mId)

        testSetList = []
        for i  in range(0,userNumMax):
            testSetList.append(list())
        for dataitem in testSet:
            uIdx=dataitem[0]-1
            mId=dataitem[1]
            testSetList[uIdx].append(mId)

        return movieFun,trainSetList,testSetList

#___推荐前n个物品___
    def recommend(self,movieFun,trainSetList,recMovieNum=REC_MOV_NUM):

        recMovList=[]
        for uIdx in range(0,userNumMax):
            u_recmList=[]
            cnt=0
            while cnt< recMovieNum:
                u_watched=trainSetList[uIdx]
                rdmNum=random.randint(0,movieNumMax-1)
                
                print(rdmNum)
                mId=movieFun[rdmNum]
                if mId not in u_watched:
                    u_recmList.append(mId)
                    cnt+=1
            recMovList.append(u_recmList)
                
        return recMovList


#___系统准确率和召回率___
    def precisionAndRecall(self,recMovList,testSetList):
        '''
        准确率=每个用户的(预测电影列表和测试集电影列表交集的数量)之和allUserSameMovNum/每个用户的(测试集电影列表数量)之和 allUserTestMovNum
        召回率=每个用户的(预测电影列表和测试集电影列表交集的数量)之和allUserSameMovNum/每个用户的(预测电影列表数量)之和 allUserPdicMovNum
        '''
        print("计算非活跃用户和活跃用户的的准确率和召回率")

        Act_allUserSameMovNum=0
        Act_allUserTestMovNum=0
        Act_allUserPdicMovNum=0
        for uIdx in range(0,userNumMax):
            u_topMList=recMovList[uIdx]
            u_testMList=testSetList[uIdx]
            IntersectionNum= len(list(set(u_topMList) & set(u_testMList)))
            Act_allUserSameMovNum+=IntersectionNum
            Act_allUserTestMovNum+=len(u_testMList)
            Act_allUserPdicMovNum+=len(u_topMList)

        Act_precision=Act_allUserSameMovNum/Act_allUserTestMovNum
        Act_reCall=Act_allUserSameMovNum/Act_allUserPdicMovNum
        
        print("<============ALLUSER===========>")
        print("准确率：{:.4f}%".format(Act_precision*100))
        print("召回率：{:.4f}%".format(Act_reCall*100))
#___系统覆盖率___
    def coverage(self,recMovList):
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
            u_topMSet=set(recMovList[uIdx])
            Act_movSet=Act_movSet | u_topMSet
            Act_allRecNum+=len(u_topMSet)
        Act_SetNum=len(Act_movSet)

        print("<===========ALLUSER=============>")
        print("相对覆盖率：{:.6f}%".format(Act_SetNum/Act_allRecNum*100))
        
if __name__=="__main__":
    rate_file=r'./data/small/ratings.csv'
    tset=itemCF(rate_file,10,30)