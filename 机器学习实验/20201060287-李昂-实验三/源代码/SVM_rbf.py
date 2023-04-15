# -*- coding: utf-8 -*- 
# @Time : 4/3/23 17:34 
# @Author : ANG

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# 读入数据
data = pd.read_csv('/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验三/ex3data2.csv')

# 构建特征和标签
X = data[['x1', 'y1']]
y = data['a']

# 构建支持向量机，使用高斯核函数
for gamma in [0.1, 1, 10, 100]:
    svc = SVC(kernel='rbf', gamma=gamma).fit(X, y)
    # 预测并输出准确率
    accuracy = svc.score(X, y)
    print(f"gamma={gamma}, Accuracy={accuracy:.3f}")

    # 可视化
    plt.figure()
    plt.title(f"gamma={gamma}")
    plt.scatter(X['x1'], X['y1'], c=y, s=50, cmap='autumn')
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()



