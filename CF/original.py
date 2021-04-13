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

    def __init__(self,rate_file):

        ratingsData=self.dataLoad(rate_file)
        trainSet,testSet=self.dataSplit(ratingsData)
        userSimilarity,userRowModvieMatrix=self.userSimFunc(trainSet)
        finalMovieRecList=self.recommend(1,userSimilarity,userRowModvieMatrix)
        precs=self.precision(testSet,finalMovieRecList,1)
        reCall=self.reCall(userRowModvieMatrix,finalMovieRecList,1)
    def dataLoad(self,rate_file):
        ###数据载入###

        # print("读取movies.csv数据...")
        # movie_file=r'C:/Users/chasu/Desktop/dataSET/small/movies.csv'   #movieId,title,genres
        # moviesData=[]
        # movieSet=set()
        # for line in open(movie_file):
        #     movieId,title,genres = line.split(",")
        #     moviesData.append((int(movieId),title))
        #     movieSet.add(movieId)

        print("读取ratings.csv数据，并处理...")
        # timest=time.time()
        ratingsData=[]
        # userSet=set()
        for line in open(rate_file):
            userid,itemid,record,idcode = line.split(",")
            # userSet.add(userid)
            ratingsData.append((int(userid),int(itemid),float(record)-std))     #data列表 [(user,movie,record),(...)...]
        # timend=time.time()
        # print("10w数据读取时间:{:.3f}".format(timend-timest))
        return ratingsData

    ###分割数据集###
    def dataSplit(self,ratingsData):
        
        # timest=time.time()
        trainSet=[]
        testSet=[]
        for dataitem in ratingsData:
            if random.randint(0,99)<testRate:
                testSet.append(dataitem)
            else:
                trainSet.append(dataitem)
        # timend=time.time()
        # print("数据分割时间:{:.3f}".format(timend-timest))
        return trainSet,testSet

    ###用户相似度矩阵###
    def userSimFunc(self,trainSet):
        # timest=time.time()
        if os.path.exists("./data/userSimilarity.npy") and os.path.exists("./data/userRowModvieMatrix.npy") :
            print("从文件加载 ...")
            userSimilarity = np.load("./data/userSimilarity.npy")
            userRowModvieMatrix=np.load("./data/userRowModvieMatrix.npy")
        else:
            print("开始计算用户相似度 ...")
            userRowModvieMatrix = np.full((userNumMax+1, movieNumMax+1), 0, dtype=float)  #生产usersNum*moviesNum大小的矩阵，默认值为std
            for dataitem in trainSet:
                userRowModvieMatrix[ dataitem[0]][ dataitem[1] ]=dataitem[2]
            #计算两个向量的内积(包括了特殊-模)
            userRowInner=np.full((userNumMax+1, userNumMax+1), 0, dtype=float)
            for i in range(1,userNumMax+1):
                for j in range(i,userNumMax+1):
                    userRowInner[i][j] =np.dot(userRowModvieMatrix[i],userRowModvieMatrix[j])
            #计算用户相似度
            userSimilarity = np.full((userNumMax+1, userNumMax+1), 0, dtype=float)
            for i in range(1,userNumMax+1):     #range 左闭右开
                for j in range(i+1,userNumMax+1):
                    userSimilarity[i][j]=userRowInner[i][j] /(math.sqrt(userRowInner[i][i])*math.sqrt(userRowInner[j][j]))
                    userSimilarity[j][i]=userSimilarity[i][j]
            print("保存文件 ...")
            np.save("./data/userSimilarity.npy",userSimilarity)
            np.save("./data/userRowModvieMatrix.npy",userRowModvieMatrix)
        return userSimilarity,userRowModvieMatrix
        # timend=time.time()
        # print("计算相似度时间矩阵:{:.3f}".format(timend-timest))
    def recommend(self,userId,userSimilarity,userRowModvieMatrix,userRfnsNum=10,movieNum=40):
        userIdFriends=userSimilarity[userId]    # userId与所有用户的相似度向量(1*userNumMax)
        friendSimilarList=[]                    # 列表，存储[userid,similar]列表，即[[userid,similar],[..]...]
        for friendId in range(1,userNumMax+1):
            friendSimilarList.append([friendId,userIdFriends[friendId]])
        topFriendList=sorted(friendSimilarList, key=lambda dic: dic[1],reverse=True)[0:userRfnsNum]
        # 按silimar关键字从大到小排序的friendSimilarList，的前userRfnsNum个
        # lambda 表达式，以列表的列表中第二个为关键字排序
        allMovList=np.full(movieNumMax+1, 0, dtype=float)
        for fid in range(0,userRfnsNum):
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
        return finalMovieRecList
        # for mov in finalMovieRecList:
        #     print("mid:{: >10},rec:{: >30}".format(mov[0],mov[1]))
    
    def precision(self,testSet,finalMovieRecList,userId):
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
    userRecommend=userCF(rate_file)




