# -*- coding: utf-8 -*- 
# @Time : 5/2/23 11:18
# @Author : ANG

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv('/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验四/数据集/ex1data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 将数据转换为张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# 定义三层神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义模型、损失函数和优化器
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.item()))

# 在测试集上评估模型
with torch.no_grad():
    predicted = net(X_test)
    test_loss = criterion(predicted, y_test)
    print('Test Loss: {:.4f}'.format(test_loss.item()))

