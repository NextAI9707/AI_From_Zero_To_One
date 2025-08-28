'''
数据预处理对于我们最终模型的效果的提高具有很重要的作用，其中缺失值处理是我们首先要考虑的问题。

数据缺失的原因也有很多种，我们在处理缺失值之前要根据不同的情况选择不同的处理方式，做到灵活运用各种方法。数据缺失的原因主要有以下两种：

随机缺失：由于采集数据不完全导致的数据缺失，这种缺失是相对随机的。
非随机缺失：业务逻辑上的缺失，可能是周期性缺失，或者在业务逻辑上分析确实存在缺失问题。
下面是几种常见的缺失值处理方法，我们需要根据实际情况选择。

🏷️不处理缺失值
优点：不用考虑缺失值的情况，直接忽悠，节省时间。
缺点：只适用于有限的模型。
这种情况只适用于一些特殊模型，如KNN、决策树、随机森林、神经网络、朴素贝叶斯、DBSCAN等。

因为上述几种类型对于数据的缺失值有很强的包容度，所以我们可以不必处理缺失值，把它当作一种类别。

🏷️删除缺失缺失值
优点：简单粗暴
缺点：当缺失值的占比较大时，可能会导致很多关键信息被忽略，可能会影响最终的模型拟合效果。
'''
import matplotlib.pyplot as plt
import missingno as mis
import pandas as pd
data=pd.read_csv("train.csv",index_col=0)
# data.info()
print(data.describe())
#矩阵图缺失值可视化
mis.matrix(data)
missing=data.isnull().sum()#统计缺失值
print(missing[missing>0])#Cabin       687,考虑直接删除，太多缺失值了
data.dropna(axis=0,how='any',subset=['Cabin'],inplace=True)
#Age         177可以考虑用平均数填充
'''
当数据近正态分布的时候，所有观测值都较好的聚集在平均值周围，这时适合填充平均值。
偏态分布和有离群点的分布，可以使用中位数填充。
名义变量更适用于填充众数，因为此种属性一般没有大小且顺序，可以填充众数。
'''
data['Age'].fillna(data['Age'].mean().values[0],inplace=True)
data['fuelType'].fillna(data['fuelType'].mode().values[0],inplace=True)
'''
平均值填充：最好验证一下数据特征是否符合正态分布，可以使用QQ图展示，或者直接一个直方图怼上去，后面我会撰写相关文章介绍相关内容。
中位数填充：有时候数据不符合正态分布，但是其基本是规律的。我们就观察其是否是某种偏态分布，因为数据是集中在中位数附近的，所以我们可以填充中位数。
众数填充：某些类别变量缺失时，我们可以使用众数填充。
其它：上述所有的填充方法一定要结合我们实际的先验知识，由于数据采样的不均衡，很有可能特征表面上看是不符合某种分布的。但是实际上这类数据我们明确的知道他就是正态分布，我们就可以使用相关手段填充。更多的，有些时候某些数据不符合正态分布时候，我们可以先对特征进行一些变化，例如对数变化等。这里就涉及到特征工程的内容，
'''
#多重插值
#考虑该数据前后类似数据的填充，适合时序特征，随着时间变化的数据的填充，这里的填充是需要我们观察特征的具体变化情况选择某种方式。
'''
DataFrame.interpolate(self, method=‘linear’, axis=0, limit=None, inplace=False, limit_direction=‘forward’, limit_area=None, downcast=None, **kwargs)

method:{“线性”，“时间”，“索引”，“值”，“最近”，“零”，“线性”，“二次”，“三次”，“重心”，“克罗格”，“多项式”，“样条”，“ piecewise_polynomial”，“ from_derivatives”，“ pchip”，“ akima”}
limit_direction:{“前进”，“后退”，“两者”}，默认为“前进”
inplace:如果可能，更新NDFrame。
'''

#建模预测
#使用不同的模型对缺失值进行预测，下面使用KNN近邻方法对缺失的值进行预测
import sklearn
import numpy as np
#创建该属性的存在部分作为训练集，不存在部分作为测试集
know_train_data=data[data.fuelType.notnull()]
unknow_train_data=data[data.fuelType.isnull()]
#选择目标属性作为训练集的y
y=know_train_data.loc[:,'fuelType']
y_train=np.array(y)
#选择其它属性作为训练集的x
x=know_train_data.loc[:,'v_1':]
x_train=np.array(x)
#在不存在的部分作为测试集
x_test = np.array(unknow_train_data.loc[:, 'v_1':])
y_test = np.array(unknow_train_data.loc[:, 'fuelType'])
#使用KNN模型进行预测，n_neighbors表示邻居个数
clf=sklearn.neighbors.KNeighborsClassifier(n_neighbors=14,weights="distance").fit(x_train,y_train)
test=clf.predict(x_test)
print(test)
#高维映射
#将属性映射到高维空间，例如某属性表示性别存在部分缺失，我们直接映射到高维表示为三个离散属性分别是，是否男，是否女，是否缺失。
#此种方法会提高我们的数据集维度，只有在样本量非常大的时候效果还好，否则会因为数据过于稀疏，效果很差。
#创建两列新变量
data['notRepairedDamage_yes'] = 0
data['notRepairedDamage_none'] = 0
data.loc[(data.notRepairedDamage.isnull()), 'notRepairedDamage_none'] = 1
data.loc[(data.notRepairedDamage.notnull()), 'notRepairedDamage_yes'] = 1
