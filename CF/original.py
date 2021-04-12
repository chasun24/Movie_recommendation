import os
import numpy as np
import random
import math
import time

class userCF:
    global std,userNumMax,movieNumMax,testRate
    std=3.0     #定义常变量表示标准，小于3表示不喜爱，大于3表示喜爱
    userNumMax=610
    movieNumMax=193609
    testRate=10     #测试集占源数据集的比率

    def __init__(self,rate_file):

        ratingsData=self.dataLoad(rate_file)
        trainSet,testSet=self.dataSplit(ratingsData)
        userSimilarity=self.userSimFunc(trainSet)
        
    
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
            ratingsData.append((int(userid),int(itemid),float(record)-std))     #data列表 ((user,movie,record),(...)...)
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
        if os.path.exists("./data/userSimilarity.npy"):
            print("用户相似度从文件加载 ...")
            userSimilarity = np.load("./data/userSimilarity.npy")
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
            print("保存计算用户相似度文件 ...")
            np.save("./data/userSimilarity.npy",userSimilarity)
        return userSimilarity
        # timend=time.time()
        # print("计算相似度时间矩阵:{:.3f}".format(timend-timest))


if __name__=="__main__":
    rate_file=r'C:/Users/chasu/Desktop/dataSET/small/ratings.csv'
    userRecommend=userCF(rate_file)




