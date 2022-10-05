import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Load the diabetes dataset
iris = datasets.load_iris()


iris_X_train , iris_X_test , iris_y_train , iris_y_test = \
    train_test_split(iris.data,iris.target,test_size=0.2)


# KMeans 演算法
kmeans  = KMeans(n_clusters = 3)
kmeans.fit(iris_X_train)
y_predict=kmeans.predict(iris_X_train)


x1=iris_X_train[:, 0]
y1=iris_X_train[:, 1]
plt.scatter(x1,y1, c=y_predict, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
