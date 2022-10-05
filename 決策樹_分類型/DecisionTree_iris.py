import matplotlib.pyplot as plt
from sklearn import tree
import DecisionTree_MyFun

# iris.xlsx 讀取後，pandas 轉  numpy 資料切割
iris_X_train, iris_X_test, iris_y_train, iris_y_test=\
              DecisionTree_MyFun.ML_read_excel("iris.xlsx",
              ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"],
              ["target"])

DecisionTree_MyFun.pyplot_中文()
DecisionTree_MyFun.ML_分類_DecisionTree(iris_X_train, iris_X_test, iris_y_train, iris_y_test)