# -*- coding: utf-8 -*- 
# @Time : 3/11/23 22:33 
# @Author : ANG

# 线性回归模型分析
# 1. 从csv文件中读取数据，并进行数据预处理（pandas）
# 2. 模型训练（sklearn）
# 3. 数据可视化(matplotlib)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 使用matplotlib绘制图像
def runplt():
    plt.figure()
    plt.title("profit plotted against population")
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.grid(True)
    plt.xlim(0, 25)
    plt.ylim(-5, 25)
    return plt


# 从scv中读取训练集
data_set = pd.read_csv("/Users/wallanceleon/Desktop/机器学习/机器学习实验/实验一/数据集/ex1data.csv")

# 划分训练集和测试集
train_set = data_set.sample(frac=0.8, random_state=0)
test_set = data_set.drop(train_set.index)

# 训练集
train_population = train_set.loc[:, 'population'].values
train_profit = train_set.loc[:, 'profit'].values

# 测试集
test_population = test_set.loc[:, 'population'].values
test_profit = test_set.loc[:, 'profit'].values

# 构造回归对象
model = LinearRegression()

x = train_population.reshape((-1, 1))
y = train_profit
model.fit(x, y)
# 获取预测值
predict_y = model.predict(x)
w = model.coef_
b = model.intercept_
print("线性回归方程为：")
print("y = ", w[0], "x + ", b)

# 显示训练集散点图和得到的回归直线
train_plot = runplt()
train_plot.plot(train_population, train_profit, 'k.')
train_plot.plot(x, predict_y, color='blue', linewidth=1)
train_plot.show()

# 显示测试集散点图和得到的回归直线
test_plot = runplt()
test_plot.plot(test_population, test_profit, 'k.')
test_plot.plot(x, predict_y, color='blue', linewidth=1)
test_plot.show()


