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
        ratingList,movieFun,activeUsrList,testSetList=self.setUser_MovieNum_MoiveFun(rate_file)
        ratingMat=self.getRatingsData(ratingList,movieFun)
        byMat=self.toByMat(ratingList,movieFun)
        Mv_MvMat=self.invertedList(byMat)
        neighborMat=self.itemNeighbors(Mv_MvMat)
        interestMat=self.itemInterest(Mv_MvMat,ratingMat,neighborMat)
        topMovList,topMovListWithSim=self.recommend(ratingMat,movieFun,interestMat)
        self.precisionAndRecall(topMovList,testSetList)
        self.coverage(topMovList)
        
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
        # 从此 movieId 可以从 movieFun中取 movieId=movieFun(movieIdx) <===> moiveIdx=movieFun(movieId) 
        # 即 可以用 mIdx 代替 mId 计算
        global userNumMax, movieNumMax
        userNumMax = len(userSet)  
        movieNumMax= len(movieSet) 

        print("计算活跃用户列表，即评论数>500的用户")
        activeUsrList=[]    # record uIdx whose ratingNums > 500
        u_ratingNum = np.full( userNumMax, 0, dtype=int)  # index:uIdx value:ratingNums
        for dataitem in ratingList:
            uIdx=dataitem[0]-1
            u_ratingNum[uIdx] +=1
        for uIdx in range(0,userNumMax):
            if u_ratingNum[uIdx]>500:
                activeUsrList.append(uIdx)

        print("数据分割为训练集和测试集")
        trainSet=[]
        testSet=[]
        for dataitem in ratingList:  
            if random.randint(0,99)<testRate:
                testSet.append(dataitem)
            else:
                trainSet.append(dataitem)

        
        testSetList = []
        for i  in range(0,userNumMax):
            testSetList.append(list())
        for dataitem in testSet:
            uIdx=dataitem[0]-1
            mId=dataitem[1]
            testSetList[uIdx].append(mId)

        return trainSet,movieFun,activeUsrList,testSetList

#___返回rating数据___
    def getRatingsData(self,ratingList,movieFun):
       
        ratingMat = np.full((userNumMax, movieNumMax), 0, dtype=float)
        for item in ratingList:
            uIdx=item[0]-1
            mIdx=movieFun.index(item[1])
            sc=item[2]
            ratingMat[uIdx][mIdx]=sc
        return ratingMat

#___转化为 uIdx-mIdx(1/0) 矩阵
    def toByMat(self,ratingList,movieFun):
        byMat = np.full((userNumMax, movieNumMax), 0, dtype=int)  
        for item in ratingList:
            uIdx=item[0]-1
            mIdx=movieFun.index(item[1])
            sc=item[2]
            if sc-std >= 0:
                byMat[uIdx][mIdx]=1
        return  byMat

#___构建所有用户物品倒排表___
    def invertedList(self,byMat):
        Mv_MvMat=np.full((movieNumMax, movieNumMax), 0, dtype=int)  
        for uId in range(0,userNumMax):
            UserList=byMat[uId]
            indexList=list(np.where(UserList==1)[0])
            uMat=self.generateSingleMat(sorted(indexList))
            Mv_MvMat+=uMat
            print("用户{}的倒排表已经生成".format(uId+1))
        return Mv_MvMat

#___生成单个用户物品倒排表___
    def generateSingleMat(self,indexList):
        uMat=np.full((movieNumMax, movieNumMax), 0, dtype=int) 
        sz=len(indexList)
        for i in range(0,sz):
            x=indexList[i]
            for j in range(i+1,sz):
                y=indexList[j]
                uMat[x][y]=1
                uMat[y][x]=1
        return uMat

#___最近邻物品___
    def itemNeighbors(self,Mv_MvMat,neighborNum=NEIGHBOR_NUM):
        neighborMat=[]
        for mIdx in range(0,movieNumMax):
            thisNeighbor=[]
            mList=Mv_MvMat[mIdx].copy()
            mi=min(mList)
            for i in range(0,neighborNum):
                mx=max(mList)
                nIdx=np.where(mList==mx)[0][0]
                thisNeighbor.append(nIdx)
                mList[nIdx]=mi
            neighborMat.append(thisNeighbor)
            print("电影{}的邻居已经找到：{}".format(mIdx,thisNeighbor))
        return neighborMat

#___用户对物品兴趣度___
    def itemInterest(self,Mv_MvMat,ratingMat,neighborMat):
        interestMat=[]
        for uIdx in range(0,userNumMax):
            uList=[]
            for mIdx in range(0,movieNumMax):
                neighborList=neighborMat[mIdx]
                u_mInterSum=0
                for nb in neighborList:
                    sim=Mv_MvMat[mIdx][nb]
                    ul=ratingMat[uIdx][nb]
                    u_mInterSum+=sim*ul
                uList.append([mIdx,u_mInterSum])
            print("用户{}对所有物品的兴趣度已经计算完成".format(uIdx))
            interestMat.append(uList)
        return interestMat

#___推荐前n个物品___
    def recommend(self,ratingMat,movieFun,interestMat,recMovieNum=REC_MOV_NUM):
        topMovList=[]
        topMovListWithSim=[]
        for uIdx in range(0,userNumMax):
            utopList=[]
            utopListWithSim=[]
            uMat=interestMat[uIdx].copy()
            uMatSorted=sorted(uMat,key=lambda x:x[1] ,reverse=True)
            cnt=0
            for item in uMatSorted:
                mIdx=item[0]
                sim=item[1]
                if ratingMat[uIdx][mIdx]==0:
                    movieId=movieFun[mIdx]
                    utopList.append(movieId)
                    utopListWithSim.append([movieId,sim])
                    cnt+=1
                    if cnt>=recMovieNum:
                        break
            topMovList.append(utopList)
            topMovListWithSim.append(utopListWithSim)
                
        return topMovList,topMovListWithSim


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
    tset=itemCF(rate_file,10,30)