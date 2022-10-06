from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.datasets import make_classification
import Random_Decision_Forests_MyFun
import matplotlib.pyplot as plt
Random_Decision_Forests_MyFun.pyplot_中文()

X=np.array([[180, 85],[174, 80],[170, 75],
      [167, 45],[158, 52],[155, 44]])
Y = np.array([1,1,1,0,0,0])

model = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=2)
model.fit(X, Y)
print("基於雜質的特徵重要性:",model.feature_importances_) # ndarray 形狀 (n_features,) 基於雜質的特徵重要性。
print("預測：",model.predict([[180,75]]))
print("準確率:",model.score(X,Y))

estimator = model.estimators_[5]  #決策樹分類器列表擬合子估計器的集合。
print("估算器",estimator)


colX = ["身高","體重"]
colY = ['man','woman']

from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree.dot',
                feature_names = colX,
                class_names =colY,
                rounded = True, proportion = False,
                precision = 2, filled = True)



#  藉由 plot 和 tree 來繪製model圖形
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 5)
for index in range(0, 5):
    tree.plot_tree(model.estimators_[index],
                   feature_names = colX,
                   class_names=colY,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')


plt.show()