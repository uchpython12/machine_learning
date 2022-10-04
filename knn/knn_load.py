import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier


# 讀取 機器學習演算法 和 權重

model = pickle.load(open("knn.model", 'rb'))

test_x=np.array([[4.6,3.1,1.5,0.2],[5,3.6,1.4,0.2,]])
test_y=np.array([0,0])
print("預測", model.predict(test_x))
print("實際", test_y)
print('準確率: %.2f' % model.score(test_x, test_y))





