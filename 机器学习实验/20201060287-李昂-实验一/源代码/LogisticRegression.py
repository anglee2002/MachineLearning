# -*- coding: utf-8 -*- 
# @Time : 3/12/23 19:59 
# @Author : ANG

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 从scv中读取训练集,并区分数据和标记
data_set = pd.read_csv("/Users/wallanceleon/Desktop/机器学习/机器学习实验/实验一/ex1data2.csv")
score = data_set.loc[:, ['Exam1', 'Exam2']].values
Is_Accepted = data_set.loc[:, 'Accepted'].values

# 使用5折交叉验证法拆分测试集、训练集
score_train, score_test, Is_Accepted_train, Is_Accepted_test = train_test_split(score, Is_Accepted, test_size=0.2,
                                                                                random_state=0)

# 训练逻辑回归模型
LogReg = LogisticRegression(solver='sag', C=1e5)
LogReg.fit(score_train, Is_Accepted_train)

a
# 显示数据集散点图
plt.figure()
plt.title("Accepted plotted against Exam1 and Exam2")
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.scatter(score[:, 0], score[:, 1], c=Is_Accepted, cmap=plt.cm.Paired)
plt.show()

# 显示训练集散点图和得到的分类结果
plt.figure()
plt.title("Accepted plotted against Exam1 and Exam2")
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.scatter(score_train[:, 0], score_train[:, 1], c=Is_Accepted_train, cmap=plt.cm.Paired)
plt.show()

# 显示测试集散点图和分类结果
plt.figure()
plt.title("Accepted plotted against Exam1 and Exam2")
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.scatter(score_test[:, 0], score_test[:, 1], c=Is_Accepted_test, cmap=plt.cm.Paired)
plt.show()
