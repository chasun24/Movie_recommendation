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
        self.recmdSys(userSimilarity,trainSetMatrix,friendsNum,movieNum)


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
        print("开始计算用户相似度 ...")
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
        timest=time.time()
        #将uid和similar整理到一个列表
        friendsList=list()  #初始化(userNumMax+1)个列表，且每个列表内为一个空列表
        for uid in range(0,userNumMax):
            friendsList.append(list())
            for fid in range(0,userNumMax):
                friendsList[uid].append(list())
                friendsList[uid][fid].append(fid)
                friendsList[uid][fid].append(userSimilarity[uid+1][fid+1])
        topFriendList=list()
        # 找到最相似的友邻
        for uid in range(0,userNumMax):
            topFriendList.append(sorted(friendsList[uid], key=lambda dic: dic[1],reverse=True)[0:friendsNum])
        # 根据相似友邻计算所有电影的推荐指数
        print("根据相似友邻计算所有电影的推荐指数")
        allMovList=list()
        for uid in range(0,userNumMax):
            allMovList.append(list())
            for mid in range(0,movieNumMax):
                allMovList[uid].append(list())
                sum=0
                for frd in topFriendList[uid]:
                    fUid=frd[0]
                    sim=frd[1]
                    sum+=sim*trainSetMatrix[fUid+1][mid+1]
                allMovList[uid][mid]=sum
            print("uID:{} 已经计算好!".format(uid))

            
        timend=time.time()
        print("推荐系统用时：{}".format(timend-timest))
        print()
#___单个用户推荐___
    def recommendForOne(self,userId,userSimilarity,userRowModvieMatrix,friendsNum,movieNum):
        print("正在计算为用户{}计算推荐电影...".format(userId))
        userIdFriends=userSimilarity[userId]    # userId与所有用户的相似度向量(1*userNumMax)
        friendSimilarList=[]                    # 列表，存储[userid,similar]列表，即[[userid,similar],[..]...]
        for friendId in range(1,userNumMax+1):
            friendSimilarList.append([friendId,userIdFriends[friendId]])
        topFriendList=sorted(friendSimilarList, key=lambda dic: dic[1],reverse=True)[0:friendsNum]
        # 按silimar关键字从大到小排序的friendSimilarList，的前friendsNum个
        # lambda 表达式，以列表的列表中第二个为关键字排序

        # 根据所有友邻计算所有电影的推荐指数
        allMovList=np.full(movieNumMax+1, 0, dtype=float)
        for fid in range(0,friendsNum):
            uid=topFriendList[fid][0]
            sim=topFriendList[fid][1]
            for mid in range(1,movieNumMax+1):
                rcd=userRowModvieMatrix[uid][mid]
                allMovList[mid]+=rcd*sim

        #所有电影的推荐指数已计算完成
        allMovieRecList=[]
        for mid in range(1,movieNumMax+1):
            allMovieRecList.append([mid,allMovList[mid]])
        finalMovieRecList=sorted(allMovieRecList, key=lambda dic: dic[1],reverse=True)[0:movieNum]
        for mov in finalMovieRecList:
            print("mid:{: >10},rec:{: >30}".format(mov[0],mov[1]))
        return finalMovieRecList


#___系统准确率和召回率___
    def precisionAndRecall(self,testSet,finalMovieRecList,userId):
        '''准确率=测试集与推荐中重合的电影/所有推荐的电影
        '''
        size=len(finalMovieRecList)
        testMatrix = np.full(movieNumMax+1, 0, dtype=float)
        for dataitem in testSet:
            if dataitem[0]!=userId:
                continue
            testMatrix[ dataitem[1] ]=dataitem[2]
        cnt=0
        for m in finalMovieRecList:
            if testMatrix[m[0]]!=0:
                cnt+=1
        print("cnt={: >10},precision={: >25}%".format(cnt,cnt/size*100))
        return cnt/size
    def reCall(self,userRowModvieMatrix,finalMovieRecList,userId):
        size=len(finalMovieRecList)
        cnt=0
        for m in finalMovieRecList:
            if userRowModvieMatrix[userId][m[0]]!=0:
                cnt+=1
        print("cnt={: >10},reCallrate={: >25}%".format(cnt,cnt/size*100))
        return cnt/size
        


if __name__=="__main__":
    rate_file=r'C:/Users/chasu/Desktop/dataSET/small/ratings.csv'
    userCF=userCF(rate_file,10,40)




