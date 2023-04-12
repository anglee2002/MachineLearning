# -*- coding: utf-8 -*-
# @Time : 3/13/23 16:53
# @Author : ANG

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LogisticRegression(object):
    """
    逻辑回归训练类
    Parameters
    ----------
    alpha : float，模型学习率
    maxiter : int，模型训练迭代次数

    """

    def __init__(self, alpha=0.3, maxiter=500):
        self.alpha = alpha  # 学习率
        self.maxiter = maxiter  # 迭代次数
        self.coef_ = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        """
        梯度提升方法训练模型特征系数
        :param x: numpy or list类型，特征变量
        :param y: numpy or list类型，目标列
        """
        x = np.mat(x)  # 将数据类型转换为numpy
        y = np.mat(y).transpose()
        m, n = np.shape(x)
        self.coef_ = np.ones((n, 1))  # 初始化特征系数（n*1）向量，[1,1,1,……]
        for k in range(self.maxiter):
            h = self.sigmoid(x * self.coef_)
            error = (y - h)
            self.coef_ = self.coef_ + self.alpha / m * x.transpose() * error  # 更新特征系数

    def predict_proba(self, x):
        """
        模型预测，返回Postive的概率
        :param x: numpy or list类型，特征变量
        :return: list类型，预测正类结果
        """
        if self.coef_ is None:
            raise ValueError('模型未进行训练')
        x = np.mat(x)  # 将数据类型转换为numpy
        return self.sigmoid(x * self.coef_)


# 从csv中读取训练集,并区分数据和标记
data_set = pd.read_csv("/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验一/数据集/ex1data2.csv")
score = data_set.loc[:, ['Exam1', 'Exam2']].values
Is_Accepted = data_set.loc[:, 'Accepted'].values

# 使用5折交叉验证法拆分测试集、训练集
score_train, score_test, Is_Accepted_train, Is_Accepted_test = train_test_split(score, Is_Accepted, test_size=0.2,
                                                                                random_state=0)

LogReg = LogisticRegression()
LogReg.fit(score_train, Is_Accepted_train)

# 预测
prepro = LogReg.predict_proba(score_test)
w = LogReg.coef_
print(w)

# 显示数据集散点图
plt.figure()
plt.title("Accepted plotted against Exam1 and Exam2")
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.scatter(score[:, 0], score[:, 1], c=Is_Accepted, cmap=plt.cm.get_cmap('viridis'))
plt.show()

# 显示训练集散点图和得到的分类结果
plt.figure()
plt.title("Accepted plotted against Exam1 and Exam2")
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.scatter(score_train[:, 0], score_train[:, 1], c=Is_Accepted_train, cmap=plt.cm.get_cmap('viridis'))
plt.show()

# 显示测试集散点图和分类结果
plt.figure()
plt.title("Accepted plotted against Exam1 and Exam2")
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.scatter(score_test[:, 0], score_test[:, 1], c=Is_Accepted_test, cmap=plt.cm.get_cmap('viridis'))
plt.show()
