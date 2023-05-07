# -*- coding: utf-8 -*- 
# @Time : 5/2/23 12:37
# @Author : ANG

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取数据集
data = pd.read_csv('/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验四/数据集/ex1data2.csv')

# 将数据集分为特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 第一层有两个输入节点，16个输出节点
        self.fc2 = nn.Linear(16, 8)  # 第二层有16个输入节点，8个输出节点
        self.fc3 = nn.Linear(8, 1)  # 第三层有8个输入节点，1个输出节点
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


net = Net()

criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 随机梯度下降优化器

for epoch in range(1000):
    inputs = torch.Tensor(X_train).float()
    labels = torch.Tensor(y_train).float()

    optimizer.zero_grad()  # 梯度清零

    outputs = net(inputs)
    loss = criterion(outputs, labels.unsqueeze(1))
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 1000, loss.item()))

# 在测试集上预测
with torch.no_grad():
    inputs = torch.Tensor(X_test).float()
    labels = torch.Tensor(y_test).float()

    outputs = net(inputs)
    predicted = torch.round(outputs)

# 计算准确率
accuracy = accuracy_score(labels, predicted)

print('Accuracy:', accuracy)
