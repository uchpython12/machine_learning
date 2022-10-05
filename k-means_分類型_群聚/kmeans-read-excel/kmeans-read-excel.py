"""
資料來源：
https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics

def ML_read_CSV(fileName, featuresCol, labelCol):
    df = pd.read_csv(fileName)  # 讀取 pandas資料
    print(df.columns)  # 印出所有列
    x = df[featuresCol]  # 設定x的資料
    print(x)  # 印出x的資料
    y = df[labelCol]  # 設定y的資料
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

def ML_群聚_KMeans(train_x, test_x, train_y, test_y,k=0):
    # KMeans 演算法
    if(k==0):
        kmeans = KMeans()
    else:
        kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)

    print("實際", test_y)
    print("預測", kmeans.predict(test_x))
    print('準確率:%.3f' % metrics.accuracy_score(test_y, kmeans.predict(test_x)))

    # Storing the predicted Clustering labels
    labels = kmeans.predict(test_x)
    # Evaluating the performance
    print("修正後的準確率: %.3f" % metrics.homogeneity_score(test_y, labels))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(test_y, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(test_y, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(test_y, labels))

    # Evaluate the score 修正答案的對照表
    hscore = metrics.homogeneity_score([0, 1, 0, 1], [1, 0, 1, 0])
    print(hscore)
    centers = kmeans.cluster_centers_
    print("中心點：",centers)
    return kmeans

train_x, test_x, train_y, test_y=\
    ML_read_CSV("Wholesale customers data.csv",
                ["Channel","Region","Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"],
                ["Channel"])
ML_群聚_KMeans(train_x, test_x, train_y, test_y)

