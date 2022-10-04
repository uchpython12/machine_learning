from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pickle

def ML_read_excel(fileName,featuresCol,labelCol):

    df=pd.read_excel(fileName)        #read pandas data
    x=df[featuresCol]          #set x data
    y=df[labelCol]             #set y data

    """ numpy X -> 2D  Y -> 1D reshape """
    x = x.to_numpy()  # x pandas to numpy
    y = y.to_numpy()  # y pandas to numpy
    print("資料筆數為", y.shape)
    # print(y)
    y = y.reshape(y.shape[0])  # y=y.to_numpy() 二維陣列,改為一維陣列
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y
    # #印出y的資料

iris_X_train, iris_X_test, iris_y_train, iris_y_test=\
              ML_read_excel("iris.xlsx",
              ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"],
              ["target"])

def ML_KNN(iris_X_train, iris_X_test, iris_y_train, iris_y_test):
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)
    print("預測", knn.predict(iris_X_train))
    print("實際", iris_y_train)
    print('準確率: %.2f' % knn.score(iris_X_train, iris_y_train))
    return knn

knn=ML_KNN(iris_X_train, iris_X_test, iris_y_train, iris_y_test)

#  save knn 演算法 和權重
pickle.dump(knn, open("knn.model", 'wb'))

"""
# 讀取 機器學習演算法 和 權重

import pickle
loaded_model = pickle.load(open("knn.model", 'rb'))
"""