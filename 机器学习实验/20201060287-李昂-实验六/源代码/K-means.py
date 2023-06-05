# -*- coding: utf-8 -*- 
# @Time : 5/29/23 16:16 
# @Author : ANG

import numpy as np
import matplotlib.pyplot as plt


def k_means(data, k, num_iterations):
    # 随机初始化聚类中心
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(num_iterations):
        # 计算每个样本点到聚类中心的距离
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        # 分配每个样本点到最近的聚类中心
        labels = np.argmin(distances, axis=1)

        # 更新聚类中心为每个簇的均值
        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis=0)

    return centroids, labels


# 读取CSV数据
data = np.genfromtxt('../数据集/ex6data.csv', delimiter=',', skip_header=1)

# 提取x和y数据
x = data[:, 0]
y = data[:, 1]

# 将x和y合并成一个二维数组
data = np.column_stack((x, y))

# 聚类的簇数和迭代次数
k = [2, 3, 4]
num_iterations = 10

# 调用 K-means 算法
for i in k:
    centroids, labels = k_means(data, i, num_iterations)
    print('k =', i, '时，聚类中心为：', centroids)
    # 绘制数据点和聚类中心
    plt.scatter(x, y, c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-means Clustering')
    plt.show()




