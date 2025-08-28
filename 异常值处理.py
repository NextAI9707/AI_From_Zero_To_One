import pandas as pd
train_data=pd.read_csv('house_data.csv')
print(train_data.describe())

#异常值又称离群点，是指那些在数据集中存在的不合理的值，需要注意的是，不合理的值是偏离正常范围的值，不是错误值。异常值出现的原因：数据集中的异常值可能是由于传感器故障、人工录入错误或异常事件导致
#异常值的检测通常有简单统计分析、3σ原则、箱型图、聚类等方法
'''
异常值的存在对数据集的影响有以下方面：

离群值严重影响数据集的均值和标准差。这些可能在统计上给出错误的结果。
它增加了误差方差，降低了统计检验的力量。
如果异常值是非随机分布的，它们会降低正态性。
大多数机器学习算法在异常值存在时不能很好地工作。因此，检测和去除异常值是很有必要的。
它们还会影响回归、方差分析和其他统计模型假设的基本假设。
'''

'''
异常值处理方法
删除含有异常值的记录：直接将含有异常值的记录删除
视为缺失值：将异常值视为缺失值，利用缺失值处理的方法进行处理
平均值修正：可用前后两个观测值的平均值修正该异常值
不处理
'''

'''
统计分析数据的最大值最小值，然后判断变量取值是否超出合理范围，
'''

'''
异常值检测原则-3o
若数据服从正态分布，则异常值被定义为一组结果值中与平均值的偏差超过三倍标准差的值。即在正态分布的假设下，距离平均值三倍σ之外的值出现的概率很小，因此可认为是异常值。
 P(|x-μ|>3δ) <= 0.003，系数3是默认，可以根据实际情况调整
'''
'''
数值分布在区间（μ-σ, μ+σ）中的概率为 0.6826
数值分布在区间（μ-2σ, μ+2σ）中的概率为 0.9545
数值分布在区间（μ-3σ, μ+3σ）中的概率为 0.9973
可以认为，数据存在随机误差，其取值几乎全部集中在（μ-3σ, μ+3σ）区间内，超出这个范围的可能性仅占不到0.3%，那么误差超过这个区间的值就识别为异常值了。
实际上，大部分真实的数据并不满足这一条件，我们就需要先对原始数据集进行Z-score变换，使用原始数据的均值（μ）和标准差（σ）进行数据的标准化。经过处理的数据也将服从标准正态分布，其均值为0，标准差为1，故3σ原则又被称为Z-score method
经过Z-score标准化后得到符合正态分布的数据，我们就可以使用3σ原则来处理数据了
'''
import matplotlib.pyplot as plt
s=train_data.B
fig=plt.figure(figsize=(6,6))
#绘制散点图
# ax1=fig.add_subplot(2,1,1)
# ax1.scatter(s.index,s.values)
# plt.grid()
# plt.show()
# 绘制密度图，使用双坐标轴
# ax2=fig.add_subplot(2,1,2)
# s.hist(bins=30,alpha=0.5,ax=ax2)
# s.plot(kind='kde',secondary_y=True,ax=ax2)
# plt.grid()
# plt.show()

#Z-zero变换
std=train_data.B.std()
mean=train_data.B.mean()
train_data['B']=train_data.B.map(lambda x:(x-mean)/std)

#3o检测异常值
mean1=0
std1=1
mark=(mean1-3*std1>train_data['B'])|(train_data['B']>mean1+3*std1)
print(train_data[mark])
'''
根据实际业务需求，若数据不服从正态分布，也可以不做标准化处理，可以用远离平均值的多少倍标准差来描述（这就使Z-score方法可以适用于不同的业务场景，只是需要根据经验来确定 kσ 中的k值，这个k值就可以认为是阈值）
'''

#异常值处理方法
'''
删除异常值
直接将含有异常值的记录删除，通常有两种策略：整条删除和成对删除。这种方法最简单简单易行，但缺点也不容忽视，一是，在观测值很少的情况下，这种删除操作会造成样本量不足；二是，直接删除、可能会对变量的原有分布造成影响，从而导致统计模型不稳定。
'''
import matplotlib.pyplot as plt
# plt.boxplot(train_data.B)
# plt.show()

train_data_outline=train_data[~mark].reset_index(drop=True)
plt.boxplot(train_data_outline.B)
plt.show()

'''
数据变换
转换变量也可以消除异常值。这些转换后的值减少了由极值引起的变化。转换方法通常有：
范围缩放：Scalling
对数变换：Log Transformation
立方根归一化：Cube Root Transformation
Box-Cox转换：Box-Cox Transformation
这些技术将数据集中的值转换为更小的值，而且不会丢失数据。如果数据有很多极端值或倾斜，数据变换有助于使您的数据正常
'''
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
result=scaler.fit_transform(train_data.B.values.reshape(-1,1))

import numpy as np
result0=np.log(train_data.B.values)
result1=np.cbrt(train_data.B.values)
import scipy
result3, maxlog = scipy.stats.boxcox(train_data.B.values ,lmbda=None)

'''
像缺失值的归责（imputation）一样，我们也可以归责异常值。在这种方法中，我们可以使用 平均值、中位数、零值 来对异常值进行替换。由于我们进行了输入，所以没有丢失数据。应选则合适的替换，这里提及一下，选择中值不受异常值的影响。以下代码仅仅作为演示。
'''
# 均值替换异常值
_mean = train_data.price.mean()
train_data[mark] = _mean
# 中位数替换异常值
_median = train_data.price.median()
train_data[mark] = _median
# 0替换异常值
train_data[mark] = 0

'''
盖帽法
也就是截尾处理，这其实也是一种插补方法。整行替换数据框里百分位数处于99%以上和1%以下的点：将99%以上的点值 = 99%的点值；小于1%的点值 = 1%的点值。
'''
'''
不处理
剔除和替换异常值或多或至少会对数据有负面影响，我们也可以根据该异常值的性质特点，使用更加稳健模型来修饰，然后直接在原数据集上进行数据挖掘。
'''

#检测一个属性的数值是否符合正态分布
'''
若随机变量x服从有个数学期望为μ,方差为σ2 的正态分布，记为N(μ,σ)
其中期望值决定密度函数的位置，标准差决定分布的幅度，当υ=0，σ=1 时的正态分布是标准正态分布
可视化分析
#导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#构造一组随机数据
s = pd.DataFrame(np.random.randn(1000)+10,columns = ['value'])

#画散点图和直方图
fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot(2,1,1)  # 创建子图1
ax1.scatter(s.index, s.values)
plt.grid()

ax2 = fig.add_subplot(2,1,2)  # 创建子图2
s.hist(bins=30,alpha = 0.5,ax = ax2)
s.plot(kind = 'kde', secondary_y=True,ax = ax2)
plt.grid()


#导入scipy模块
from scipy import stats
"""
kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
结果返回两个值：statistic → D值，pvalue → P值
p值大于0.05，为正态分布
H0:样本符合  
H1:样本不符合 
如何p>0.05接受H0 ,反之 
"""
u = s['value'].mean()  # 计算均值
std = s['value'].std()  # 计算标准差
stats.kstest(s['value'], 'norm', (u, std))
'''
