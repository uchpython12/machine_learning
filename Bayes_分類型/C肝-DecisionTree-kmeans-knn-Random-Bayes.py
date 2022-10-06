# Load the diabetes dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
import Bayes_Myfun
# Load the diabetes dataset
Bayes_Myfun.pyplot_中文()
print("====讀取資料==============")

df = pd.read_excel("C肝.xlsx",0) #('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(df.columns)

colX=['Age', 'Sex' , 'ALB', 'ALP',   'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
colY=['Category']
colY2=df['Category'].unique().tolist()  #  文字轉數字

print(df.head())
X=df[colX]
X=np.array(X)
Y=df[colY]
Y=np.array(Y)
Y=Y.ravel()     # 2D　轉1D

print(" X shape",X.shape)
print(" Y shape",Y.shape)



print("====資料拆分==============")
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.02)

print("實際的答案           ：",test_y)


# Bayes


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y.ravel())
prediction = model.predict(test_x)
modelScore=model.score(test_x,test_y)
print("Naive Bayes 預估答案：",prediction," 準確率：",modelScore)

# 隨機森林

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=2)
rf.fit(train_x, train_y.ravel())
prediction = rf.predict(test_x)
rfScore=rf.score(test_x,test_y)
print("隨機森林 預估答案      ：",prediction," 準確率：",rfScore)
###
from sklearn.tree import export_graphviz
export_graphviz(rf.estimators_[2], out_file='隨機森林1.dot',
                feature_names = colX,
                # class_names =colY2,
                rounded = True, proportion = False,
                precision = 2, filled = True)


#######
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 5)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = colX,
                   #class_names=colY2,
                   filled = True,
                   ax = axes[index])

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('隨機森林1.png')
plt.show()


# 決策樹
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x,train_y.ravel())
prediction = clf.predict(test_x)
clfScore=clf.score(test_x,test_y)
print("決策樹 預估答案       ：",prediction," 準確率：",clfScore)


#####
tree.export_graphviz(clf,out_file='決策樹.dot')
fig = plt.figure()
_ = tree.plot_tree(clf,
                   feature_names = colX,
                   #class_names=colY2,
                   filled=True)
fig.savefig("決策樹1.png")
plt.show()




# KMeans 演算法
kmeans  = KMeans(n_clusters = 3)
kmeans.fit(train_x)
y_predict=kmeans.predict(test_x)
kmeansScore = metrics.accuracy_score(test_y,kmeans.predict(test_x))
print("KMeans 演算法 預估答案：",y_predict," 準確率：",kmeansScore)

# KNN 演算法

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=1)
knn.fit(train_x, train_y)
knnPredict = knn.predict(test_x)
knnScore=knn.score(test_x, test_y)
print("KNN    演算法 預估答案：",knnPredict," 準確率：",knnScore)

# 決策樹 演算法
from sklearn import tree
import pydot
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(train_x, train_y)
tree.export_graphviz(clf,out_file='tree-C1.dot')
clfPredict = clf.predict(test_x)
clfScore1=clf.score(test_x, test_y)
print("決策樹 1 演算法 預估答案：",clfPredict," 準確率：",clfScore1)



# 決策樹 演算法

clf = tree.DecisionTreeClassifier( criterion='entropy',splitter='random',max_depth=2)
clf = clf.fit(train_x, train_y)
tree.export_graphviz(clf,out_file='tree-C2.dot')
clfPredict = clf.predict(test_x)
clfScore2=clf.score(test_x, test_y)
print("決策樹 2 演算法 預估答案：",clfPredict," 準確率：",clfScore2)


#######
fig = plt.figure()
_ = tree.plot_tree(clf,
                   feature_names=colX,
                   #class_names=colY,
                   filled=True)

fig.savefig("decistion_tree.png")
# plt.show()
