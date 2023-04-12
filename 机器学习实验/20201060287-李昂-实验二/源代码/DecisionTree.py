# -*- coding: utf-8 -*- 
# @Time : 3/27/23 16:07 
# @Author : ANG

import pandas as pd
import sklearn.tree
from sklearn.model_selection import train_test_split, GridSearchCV
import pydotplus
import pprint


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

    # 将70%的数据用作训练集，30%的数据用作测试集
    sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.3, random_state=0)

    return sample_train, sample_test, label_train, label_test


def DecisionTree():
    """
        本函数用于决策树模型的训练，具体说明如下：

        :param

        criterion：特征选择标准 【entropy, gini】
                   默认gini，即CART算法

        splitter： 特征划分标准 【best, random】
                   best在特征的所有划分点中找出最优的划分点，random随机的在部分划分点中找局部最优的划分点
                   默认的‘best’适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐‘random’

        max_depth：决策树最大深度 【int,  None】
                   默认值是‘None’
                   一般数据比较少或者特征少的时候可以不用管这个值
                   如果模型样本数量多，特征也多时，推荐限制这个最大深度,具体取值取决于数据的分布
                   常用的可以取值10-100之间，常用来解决过拟合

        min_samples_split：内部节点（即判断条件）再划分所需最小样本数 【int, float】
                           默认值为2。如果是int，则取传入值本身作为最小样本数
                           如果是float，则取ceil(min_samples_split*样本数量)作为最小样本数（向上取整）

        min_samples_leaf：叶子节点（即分类）最少样本数
                          如果是int，则取传入值本身作为最小样本数
                          如果是float，则取ceil(min_samples_leaf*样本数量)的值作为最小样本数
                          这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝

        min_weight_fraction_leaf：叶子节点（即分类）最小的样本权重和 【float】
                                  这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝
                                  默认是0，就是不考虑权重问题，所有样本的权重相同
                                  一般来说如果我们有较多样本有缺失值或者分类树样本的分布类别偏差很大，就会引入样本权重，这时就要注意此值

        max_features：在划分数据集时考虑的最多的特征值数量 【int值】
                      在每次split时最大特征数；【float值】表示百分数，即（max_features*n_features）

        random_state：【int, randomSate instance, None】 默认是None

        max_leaf_nodes：最大叶子节点数 【int, None】
                        通过设置最大叶子节点数，可以防止过拟合
                        默认值None，默认情况下不设置最大叶子节点数
                        如果加了限制，算法会建立在最大叶子节点数内最优的决策树
                        如果特征不多，可以不考虑这个值，但是如果特征多，可以加限制，具体的值可以通过交叉验证得到

        min_impurity_decrease：节点划分最小不纯度 【float】
                               默认值为‘0’ 限制决策树的增长
                               节点的不纯度（基尼系数，信息增益，均方差，绝对差）必须大于这个阈值，否则该节点不再生成子节点

        min_impurity_split（已弃用）：信息增益的阀值
                                    决策树在创建分支时，信息增益必须大于这个阈值，否则不分裂
                                   （从版本0.19开始不推荐使用：min_impurity_split已被弃用，以0.19版本中的min_impurity_decrease取代)
                                    (min_impurity_split的默认值将在0.23版本中从1e-7变为0，并且将在0.25版本中删除, 请改用min_impurity_decrease）

        class_weight：类别权重 【dict, list of dicts, balanced】
                      默认为None（不适用于回归树，sklearn.tree.DecisionTreeRegressor）
                      指定样本各类别的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别
                      balanced，算法自己计算权重，样本量少的类别所对应的样本权重会更高。如果样本类别分布没有明显的偏倚，则可以不管这个参数

        presort：bool，默认是False，表示在进行拟合之前，是否预分数据来加快树的构建
                 对于数据集非常庞大的分类，presort=true将导致整个分类变得缓慢
                 当数据集较小，且树的深度有限制，presort=true才会加速分类

        ccp_alpha：将选择成本复杂度最大且小于ccp_alpha的子树
                   默认情况下，不执行修剪

        :return: 决策树模型

        """
    decision_tree = sklearn.tree.DecisionTreeClassifier()
    return decision_tree


def DecisionTree_self_adjust():
    """
    根据情景需求，手动调节参数：
    本次数据为离散数据，而entropy对离散数据的处理效果更好，所以选择entropy作为特征选择标准；
    min_samples_split：4，即内部节点再划分所需最小样本数为4，防止过拟合；
    min_samples_leaf：2，即叶子节点最少样本数为2，防止过拟合；
    :return: 决策树模型

    """
    decision_tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=4, min_samples_leaf=2)
    return decision_tree


def DecisionTree_adjust_by_gridsearch():
    """
    通过网格搜索调整参数
    :return: 决策树模型

    """
    decision_tree = sklearn.tree.DecisionTreeClassifier()
    parameters = {'criterion': ('gini', 'entropy'),
                  'splitter': ('best', 'random'),
                  'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                  }

    decision_tree = GridSearchCV(decision_tree, param_grid=parameters, return_train_score=True)
    return decision_tree


def main():
    """
    此函数用于测试决策树模型
    :return: 测试结果

    """

    sample_train, sample_test, label_train, label_test = loaddata(
        "/Users/wallanceleon/Desktop/机器学习/机器学习实验/20201060287-李昂-实验二/Dataset/ex2data.csv")

    # 生成决策树模型
    decision_tree = DecisionTree()
    decision_tree.fit(sample_train, label_train)

    # 生成主观调整参数后的决策树模型
    decision_tree_self_adjust = DecisionTree_self_adjust()
    decision_tree_self_adjust.fit(sample_train, label_train)

    # 生成通过网格搜索调整参数后的决策树模型
    decision_tree_adjust_by_gridsearch = DecisionTree_adjust_by_gridsearch()
    decision_tree_adjust_by_gridsearch.fit(sample_train, label_train)

    # 可视化决策树
    dot_data = sklearn.tree.export_graphviz(decision_tree, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("decision_tree.pdf")

    # 可视化根据训练集主观调整参数后的决策树
    dot_data = sklearn.tree.export_graphviz(decision_tree_self_adjust, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("decision_tree_self_adjust.pdf")

    # 可视化通过网格搜索调整参数后的决策树
    decision_tree_adjust_by_gridsearch = decision_tree_adjust_by_gridsearch.best_estimator_
    dot_data = sklearn.tree.export_graphviz(decision_tree_adjust_by_gridsearch, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("decision_tree_adjust_by_gridsearch.pdf")

    # 打印决策树的最优参数
    print('最优参数为:')
    pprint.pprint(decision_tree_adjust_by_gridsearch.get_params())

    # 评估模型
    print("未剪枝和预剪枝的决策树模型：")
    print("决策树模型的准确率为：", decision_tree.score(sample_test, label_test))
    print("根据训练集主观调整参数进行预剪枝的决策树的准确率为：",
          decision_tree_self_adjust.score(sample_test, label_test))
    print("通过网格搜索调整参数进行预剪枝的决策树的准确率为：",
          decision_tree_adjust_by_gridsearch.score(sample_test, label_test))


if __name__ == '__main__':
    main()
