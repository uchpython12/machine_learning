import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy

#read data
dataframe = pd.read_fwf('brain_body.txt')    # 讀取　txt
x_values = dataframe[['Body']]               # 以體重為主
y_values = dataframe[['Brain']]              # 腦子重量

#訓練
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#預測
pre=body_reg.predict(x_values)

import Myfun
Myfun.pyplot_中文()

#圖形化

plt.scatter(x_values, y_values)
x_values2=numpy.array(x_values)
plt.plot(x_values2,pre, "r--")

plt.xlabel('體重')
plt.ylabel('腦子重量')
plt.title("體重 和 腦子")
plt.show()
