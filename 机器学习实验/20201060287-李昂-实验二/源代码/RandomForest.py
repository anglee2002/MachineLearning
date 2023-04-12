# -*- coding: utf-8 -*- 
# @Time : 3/26/23 20:31
# @Author : ANG

import pandas as pd
import sklearn.ensemble as ensemble
from sklearn.model_selection import cross_val_score, GridSearchCV


def loaddata(path):
    """
    本函数用于数据预处理,为训练决策树模型做准备,对于各类数据的具体处理如下:

    PassengerId（乘客ID），Name（姓名），Ticket（船票信息），Cabin（船舱）对于是否存活意义不大，不加入后续的分析；
    Survived（获救情况）变量为因变量，其值只有两类1或0，代表着获救或未获救,保持不变；
    Pclass（乘客等级），Sex（性别），Embarked（登船港口）是明显的类别型数据,保持不变；
    Age（年龄），SibSp（堂兄弟妹个数），Parch（父母与小孩的个数）是隐性的类别型数据,保持不变；
    Fare（票价）是数值型数据；Cabin（船舱）为文本型数据,保持不变；
    Age（年龄），和Embarked（登船港口）信息存在缺失数据,进行填充处理；

    :param path: 所需数据的绝对路径
    :return: DataFrame格式的数据列表

    """
    dataset = pd.read_csv(path, index_col=None)

    # 删除无用的列
    dataset.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    dataset.drop(["Cabin"], axis=1, inplace=True)

    # 将Sex（性别）列的值转换为数值型数据
    dataset["Sex"].replace({"male": 0, "female": 1}, inplace=True)

    # 将Embarked（登船港口）列的值转换为数值型数据
    dataset["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

    # 处理缺失值
    # Age（年龄）缺失值用平均值填充
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    # Embarked（登船港口）缺失值用众数填充
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # 划分数据集和标签集
    sample = dataset.iloc[:, 1:]
    label = dataset.iloc[:, 0]

    return sample, label


def RandomForest(sample, label):
    """
    本函数用于训练随机森林模型,并对模型进行评估,输出模型的评估结果
    :param sample: 数据集
    :param label: 标签集
    :return: randomforest模型

    """
    random_forest = ensemble.RandomForestClassifier()
    # 使用10折交叉验证模型评估，输出每次验证的准确率和平均准确率
    scores = cross_val_score(random_forest, sample, label, cv=10)
    print("随机森林模型的平均准确率为：", scores.mean())


def RandomForest_self_adjust(sample, label):
    """
    本函数用于调整随机森林模型的参数,并对模型进行评估,输出模型的评估结果
    对于随机森林模型的参数调整,主要调整的参数有:
    n_estimators：森林中决策树的数量，默认100
                 表示这是森林中树木的数量，即基基评估器的数量
                 这个参数对随机森林模型的精确性影响是单调的n_estimators越大，模型的效果往往越好
                 但是相应的，任何模型都有决策边界n_estimators达到一定的程度之后，随机森林的精确性往往不在上升或开始波动
                 并且，n_estimators越大，需要的计算量和内存也越大，训练的时间也会越来越长
                 对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡

    criterion：分裂节点所用的标准，可选“gini”, “entropy”，默认“gini”

    max_depth：树的最大深度
               如果为None，则将节点展开，直到所有叶子都是纯净的(只有一个类)
               或者直到所有叶子都包含少于min_samples_split个样本。默认是None

    min_samples_split：拆分内部节点所需的最少样本数
                       如果为int，则将min_samples_split视为最小值
                       如果为float，则min_samples_split是一个分数，而ceil（min_samples_split * n_samples）是每个拆分的最小样本数，默认是2

    min_samples_leaf：在叶节点处需要的最小样本数
                     仅在任何深度的分割点在左分支和右分支中的每个分支上至少留下min_samples_leaf个训练样本时，才考虑
                     这可能具有平滑模型的效果，尤其是在回归中。如果为int，则将min_samples_leaf视为最小值
                     如果为float，则min_samples_leaf是分数，而ceil（min_samples_leaf * n_samples）是每个节点的最小样本数，默认是1

    min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数
                              如果未提供sample_weight，则样本的权重相等

    max_features：寻找最佳分割时要考虑的特征数量：
                  如果为int，则在每个拆分中考虑max_features个特征
                  如果为float，则max_features是一个分数，并在每次拆分时考虑int（max_features * n_features）个特征
                  如果为“auto”，则max_features = sqrt（n_features）
                  如果为“ sqrt”，则max_features = sqrt（n_features）
                  如果为“ log2”，则max_features = log2（n_features）
                  如果为None，则max_features = n_features
                  注意：在找到至少一个有效的节点样本分区之前，分割的搜索不会停止，即使它需要有效检查多个max_features功能也是如此

    max_leaf_nodes：最大叶子节点数，整数，默认为None

    min_impurity_decrease：如果分裂指标的减少量大于该值，则进行分裂

    min_impurity_split：决策树生长的最小纯净度。默认是0。

    bootstrap：是否进行bootstrap操作，bool，默认True
               如果bootstrap==True，将每次有放回地随机选取样本，只有在extra-trees中，bootstrap=False

    oob_score：是否使用袋外样本来估计泛化精度，默认False

    n_jobs：并行计算数，默认是None

    random_state：控制bootstrap的随机性以及选择样本的随机性

    verbose：在拟合和预测时控制详细程度，默认是0

    class_weight：每个类的权重，可以用字典的形式传入{class_label: weight}
                  如果选择了“balanced”，则输入的权重为n_samples / (n_classes * np.bincount(y))

    ccp_alpha：将选择成本复杂度最大且小于ccp_alpha的子树
               默认情况下，不执行修剪

    max_samples：如果bootstrap为True，则从X抽取以训练每个基本分类器的样本数
                 如果为None（默认），则抽取X.shape [0]样本
                 如果为int，则抽取max_samples样本
                 如果为float，则抽取max_samples * X.shape [0]个样本

    :param sample: 数据集
    :param label: 标记集
    :return: randomforest模型

    """
    random_forest = ensemble.RandomForestClassifier(criterion='entropy', max_features='sqrt')
    # 使用10折交叉验证模型评估，输出每次验证的准确率和平均准确率
    scores = cross_val_score(random_forest, sample, label, cv=10)
    print("主观调整参数后的随机森林模型的平均准确率为：", scores.mean())


def RandomForest_adjust_by_gridsearch(sample, label):
    """
    使用网格搜索调整随机森林模型参数
    :param sample: 数据集
    :param label: 标记集
    :return: 调整后的随机森林模型

    """
    # 定义网格搜索的参数
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }
    # 定义随机森林模型
    random_forest = ensemble.RandomForestClassifier()
    # 使用网格搜索调整模型参数
    grid_search = GridSearchCV(random_forest, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(sample, label)
    print("经过网格搜索调整参数后的随机森林模型的平均准确率为：", grid_search.best_score_)


    


def main():
    """
    主函数,调用上述函数,生成随机森林模型
    :return:

    """
    sample, label = loaddata(
        "/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验二/Dataset/ex2data.csv")

    RandomForest(sample, label)

    RandomForest_self_adjust(sample, label)

    RandomForest_adjust_by_gridsearch(sample, label)


if __name__ == '__main__':
    main()
