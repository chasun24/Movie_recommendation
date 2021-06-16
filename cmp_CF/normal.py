import numpy as np


def Normalize(Mat):
    mx = np.max(Mat)
    mn = np.min(Mat)
    return [(float(j) - mn) / (mx - mn) for i in Mat for j in i]

a=[[1,2,3],[2,3,4],[3,4,5]]
b=Normalize(a)
print(b)