from sklearn import tree

# Load the diabetes dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

# Load the diabetes dataset

print("====讀取資料==============")

df = pd.read_excel("C肝.xlsx",0) #('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(df.columns)

colX=['Age', 'Sex' , 'ALB', 'ALP',   'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
colY=['Category']

print(df.head())
X=df[colX]
X=np.array(X)
Y=df[colY]
Y=np.array(Y)
Y = Y.reshape(Y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列

print("====資料拆分==============")
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.05)

print("實際的答案           ：",test_y)

# 決策樹
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
prediction = clf.predict(test_x)
clfScore=clf.score(test_x,test_y)
print("決策樹 預估答案       ：",prediction," 準確率：",clfScore)

# KMeans 演算法
kmeans  = KMeans(n_clusters = 3)
kmeans.fit(train_x)
y_predict=kmeans.predict(test_x)
kmeansScore = metrics.accuracy_score(test_y,kmeans.predict(test_x))
kmeanshomogeneity_score= metrics.homogeneity_score(test_y,kmeans.predict(test_x))
print("KMeans 演算法 預估答案：",y_predict," 準確率：",kmeansScore)
print("KMeans 演算法 預估答案：",y_predict," 修正後準確率：",kmeanshomogeneity_score)

# KNN 演算法

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=1)
knn.fit(train_x, train_y)
knnPredict = knn.predict(test_x)
knnScore=knn.score(test_x, test_y)
print("KNN    演算法 預估答案：",knnPredict," 準確率：",knnScore)

# 決策樹 演算法
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
tree.export_graphviz(clf,out_file='tree-C.dot')
clfPredict = clf.predict(test_x)
clfScore=clf.score(test_x, test_y)
print("決策樹  演算法 預估答案：",clfPredict," 準確率：",clfScore)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf,
                   feature_names=colX,
                   #class_names=colY,
                   filled=True)

fig.savefig("decistion_tree.png")
plt.show()