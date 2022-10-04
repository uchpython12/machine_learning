import pandas as pd
import numpy as np

print("read data...")

df1=pd.DataFrame({
    "x":[1,2,3,4,5],
    "y":[51,52,53,54,55],
    "z":[1,2,3,4,5],
    "c":[0,0,0,1,1],
})
x=df1[["x","y","z"]].to_numpy()
y=df1[["c"]].to_numpy()
print(x)

from sklearn.preprocessing import MinMaxScaler

def ML_標準化1_轉換(x):
    print("===標準化===")
    scaler = MinMaxScaler()  # 初始化
    scaler.fit(x)  # 找標準化範圍
    x1 = scaler.transform(x)  # 把資料轉換
    return x1,scaler

x1,scaler = ML_標準化1_轉換(x)
print(x1)

def ML_標準化1_還原(x1):
    print("還原")
    x2 = scaler.inverse_transform(x1)
    return x2
x2 = ML_標準化1_還原(x1)
print(x2)

print("====標準化 方法2==============")


def ML_標準化2_轉換(x):
    dict1={}
    min1=np.min(x,axis=0)
    dict1["min"]=min1
    max1=np.max(x,axis=0)
    dict1["max"]=max1
    dist=max1-min1
    dict1["dist"]=dist
    #  x-最低/(最高-最低)
    x2=(x-min1)/dist
    return x2,dict1

x2,dict1=ML_標準化2_轉換(x)
print(x2)
def ML_標準化2_還原(x1,dict1):
    min1=dict1["min"]
    max1=dict1["max"]
    dist=dict1["dist"]
    #  (x1*(最高-最低))+最低
    x2 = (x1*(dist))+min1
    return x2

x3=ML_標準化2_還原(x2,dict1)
print(x3)




