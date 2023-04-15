# -*- coding: utf-8 -*- 
# @Time : 4/3/23 16:12 
# @Author : ANG


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    数据预处理
    :return: 特征数据和标签数据
    """
    data = pd.read_csv(path)
    # 将第一列和第二列的数据作为特征
    x = data.loc[:, ['x1', 'y1']].values
    # 将第三列的数据作为标签
    label = data.loc[:, ['a']].values

    # 将数据集分为训练集和测试集
    x_train, x_test, label_train, label_test = train_test_split(x, label, test_size=0.3, random_state=0)

    return x_train, x_test, label_train, label_test


x_train, x_test, label_train, label_test = load_data(
    '/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验三/ex3data1.csv')


def SVM_linear_kernal(x_train, x_test, label_train, label_test):
    """
    SVM线性核函数
    :param x_train: 训练集特征
    :param x_test: 测试集特征
    :param label_train: 训练集标签
    :param label_test: 测试集标签
    :return: None
    """
    # 创建SVM分类器
    clf = svm.SVC(kernel='linear')

    # 尝试不同的惩罚系数C进行训练和测试
    for C in [0.1, 1, 10]:
        # 设定惩罚系数C
        clf.set_params(C=C)
        # 训练
        clf.fit(x_train, label_train)
        # 预测
        y_pred = clf.predict(x_test)
        # 评估模型性能
        train_score = clf.score(x_train, label_train)
        test_score = clf.score(x_test, label_test)
        print(f"C={C}, train score={train_score:.3f}, test score={test_score:.3f}")
        # 依据训练集画出决策边界
        plt.figure()
        plt.title(f'C={C}')
        # 限定x轴和y轴的范围
        plt.xlim(0, 5)
        plt.ylim(1, 5)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=label_train, s=20, cmap=plt.cm.Paired)
        plt.xlabel('x1')
        plt.ylabel('y1')
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        plt.show()

        # 画出不同参数对测试集的分裂效果
        plt.figure()
        plt.title(f'C={C}')
        # 限定x轴和y轴的范围
        plt.xlim(0, 5)
        plt.ylim(1, 5)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=label_test, s=20, cmap=plt.cm.Paired)
        plt.xlabel('x1')
        plt.ylabel('y1')
        # 画出决策边界
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        plt.show()


SVM_linear_kernal(x_train, x_test, label_train, label_test)
