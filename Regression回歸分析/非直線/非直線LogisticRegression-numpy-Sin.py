
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import Myfun
Myfun.pyplot_中文()


rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x)+5 + 0.1 * rng.randn(50)
plt.scatter(x, y,label="實際資料")

# 訓練
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
t1=x[:, np.newaxis]               # 1D轉 2D
poly_model.fit(t1, y)

# 畫出預測
xfit = np.linspace(0, 10, 1000)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.plot(xfit, yfit,label="畫出預測")

plt.show()