from sklearn.cluster import KMeans
import numpy as np

x = np.array([[1, 2], [1, 4], [1, 0],[10,2],[10,4],[10,0], [20,20],[20,21],[20,22],[20,19],[20,18],[20,17]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

print(kmeans.labels_)

# 预测
y_pre = kmeans.predict([[0, 0], [12, 3]])
print(y_pre)

# 聚类中心
print(kmeans.cluster_centers_)