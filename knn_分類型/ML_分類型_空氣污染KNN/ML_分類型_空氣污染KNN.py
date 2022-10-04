from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def ML_read_excel(fileName,featuresCol,labelCol):
    df=pd.read_excel(fileName)        #讀取 pandas資料
    print(df.columns)                    #印出所有列
    x=df[featuresCol]          #設定x的資料
    print(x)                                                                                       #印出x的資料
    y=df[labelCol]                                                                               #設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y
    # #印出y的資料


iris_X_train, iris_X_test, iris_y_train, iris_y_test=\
              ML_read_excel("空氣污染.xlsx",
              ["pollutant","so2","co","o3","o3_8hr","pm10","pm2.5","no2","nox",
               "no","wind_speed","wind_direc","co_8hr","pm2.5_avg",
               "pm10_avg","so2_avg"],
              ["target2"])
def ML_分類_KNN(iris_X_train, iris_X_test, iris_y_train, iris_y_test,k=5,p=1):

    knn = KNeighborsClassifier(n_neighbors=k,p=p)
    knn.fit(iris_X_train, iris_y_train)
    print("預測", knn.predict(iris_X_test))
    print("實際", iris_y_test)
    print('準確率: %.2f' % knn.score(iris_X_test, iris_y_test))
    return knn

knn= ML_分類_KNN(iris_X_train, iris_X_test, iris_y_train, iris_y_test)