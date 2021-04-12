# -*- coding: utf-8 -*-
"""
Author: yc
Desc: 一个基于UserCF算法的电影推荐系统

"""

import random
import math
import json
import os

class UserCFRec:
    def __init__(self,datafile):
        self.datafile = datafile
        self.data = self.loadData()

        self.trainData,self.testData = self.splitData(30)  # 训练集与测试集
        self.users_sim = self.UserSimilarityBest()
        
    # 加载评分数据到data
    def loadData(self):
        print("加载数据...")
        data=[]
        for line in open(self.datafile):
            userid,itemid,record,_ = line.split(",")
            data.append((userid,itemid,int(float(record))))
              
        return data

    """
        拆分数据集为训练集和测试集
            k: 参数
            seed: 生成随机数的种子
            M: 随机数上限
    """
    def splitData(self,k):     # k=3 seed =47
        print("训练数据集与测试数据集切分...")
        train,test = {},{}
        testCount=0
        trainCount=0
        for user,item,record in self.data:
            # 1/9
            #返回 0~M之间的随机一个数,双闭区间

            if random.randint(0,99) <k:        
                test.setdefault(user,{})
                test[user][item] = record
                # testCount+=1
            else:
                train.setdefault(user,{})
                train[user][item] = record
                # trainCount+=1

        # print("trainCount:{},testCount:{}".format(trainCount,testCount))
        # print("train:{},test:{}".format(len(train),len(test)))
        # print("train/all={},test/all={}".format(trainCount/(testCount+trainCount),testCount/(trainCount+testCount)))
        return train,test

    # 计算用户之间的相似度，采用惩罚热门商品和优化算法复杂度的算法
    def UserSimilarityBest(self):
        print("开始计算用户之间的相似度 ...")
        if os.path.exists("data/user_sim.json"):
            print("用户相似度从文件加载 ...")
            userSim = json.load(open("data/user_sim.json","r"))
        else:
            # 得到每个item被哪些user评价过
            item_users = dict()         #函数用于创建一个字典。
            for u, items in self.trainData.items():     #self.trainData.items是字典中的每一个项（key&&value）
                for i in items.keys():          #items.keys()是 {"user",{"movie","record"}} 中的movie
                    # 将item_users字典添加所有的key，并设置value为空集合，存放用户
                    item_users.setdefault(i,set())      #和 get()方法 类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
                    if self.trainData[u][i] > 0:        #若record不为0
                        item_users[i].add(u)            #则加入用户至集合中，形成倒排表{"movie",("user1","user2",...)}
            count = dict()
            user_item_count = dict()
            for i, users in item_users.items():     #{"movie",("user1","user2",...)}
                for u in users:
                    user_item_count.setdefault(u,0) #无则初始化，有则不处理
                    user_item_count[u] += 1         #字典中的user对应++
                    count.setdefault(u,{})          #将user以key加入count
                    for v in users:
                        count[u].setdefault(v, 0)   #对于每个u-u构建相似度矩阵，且初始化为0
                        if u == v:                  # u-u即为0
                            continue
                        count[u][v] += 1 / math.log(1+len(users))       # 非u-u
            # 构建相似度矩阵
            userSim = dict()
            for u, related_users in count.items():
                userSim.setdefault(u,{})
                for v, cuv in related_users.items():
                    if u==v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
            json.dump(userSim, open('data/user_sim.json', 'w'))
        return userSim

    """
        为用户user进行物品推荐
            user: 为用户user进行推荐
            k: 选取k个近邻用户
            nitems: 取nitems个物品
    """
    def recommend(self, user, k=8, nitems=40):
        result = dict()
        have_score_items = self.trainData.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    """
        计算准确率
            k: 近邻用户数
            nitems: 推荐的item个数
    """
    def precision(self, k=8, nitems=10):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        for user in self.trainData.keys():
            tu = self.testData.get(user, {})
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)

if __name__=='__main__':
    cf = UserCFRec(".\\data\\ml-1m\\ratings.csv")
    result = cf.recommend("1")
    print("user '1' recommend result is {} ".format(result))

    precision = cf.precision()
    print("precision is {}".format(precision))





