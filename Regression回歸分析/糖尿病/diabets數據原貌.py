import matplotlib.pyplot as plt  #繪製函式庫
import numpy as np #矩陣函式庫
from sklearn import datasets, linear_model #線性回歸函式庫
import Myfun
Myfun.pyplot_中文()
# 取得糖尿病的數據
diabetes = datasets.load_diabetes()   # 讀取　糖尿病
# 取得
diabetes_X = diabetes.data[:, np.newaxis, 2]  # 只取 第3個特徵值BMI

# 切分特徵值BMI
diabetes_X_train = diabetes_X[:-20]          # 扣掉最後20筆 為訓練數據
diabetes_X_test = diabetes_X[-20:]           # 最後20筆 為測試數據

# 切分答案
diabetes_y_train = diabetes.target[:-20]    # 扣掉最後20筆 為訓練數據 答案
diabetes_y_test = diabetes.target[-20:]     # 最後20筆 為測試數據 答案

# 繪圖
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.xlabel('BMI')
plt.ylabel('糖尿病機率')
plt.show()