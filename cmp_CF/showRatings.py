import matplotlib.pyplot as plt

import numpy as np
userNumMax=610
movieNumMax=193609

rate_file=r'./data/small/ratings.csv'
print("从文件读入数据")
ratingsData=[]
file_open= open(rate_file)
for line in file_open:
    userid,itemid,record,idcode = line.split(",")
    ratingsData.append((int(userid),int(itemid),float(record)))     #data列表 [(user,movie,record),(...)...]

u_ratingNum = np.full( userNumMax, 0, dtype=int)  # index:uIdx value:ratingNums
for dataitem in ratingsData:
    uIdx=dataitem[0]-1
    u_ratingNum[uIdx] +=1


maxValue=max(u_ratingNum)   # the biggest rating number 
bar=np.full( maxValue+1, 0, dtype=int)  # index:ratingNums  value:userNum   [0~maxValue]
simpeBar=np.full( 6, 0, dtype=int)  # index:ratingNums  value:userNum   
for value in u_ratingNum:
    bar[value]+=1
    if value < 500:
        simpeBar[0]+=1
    elif value < 1000:
        simpeBar[1]+=1
    elif value < 1500:
        simpeBar[2]+=1
    elif value <2000:
        simpeBar[3]+=1
    elif value <2500:
        simpeBar[4]+=1
    else :
        simpeBar[5]+=1



# y=bar
# x=range(0,maxValue+1)
# plt.plot(x,y)
# plt.show()
plt.xlabel('ratingNums')
plt.ylabel('userNums')
plt.title(u'user-ratingNumbers-Histogram')
simpe_X=[0,500,1000,1500,2000,2500,3000]
plt.hist(u_ratingNum,simpe_X)
plt.show()
print("ok")