import pandas as pd
import numpy as np
# 划分数据集用到的
from sklearn .model_selection  import train_test_split
# 特征工程用到的
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# 逻辑回归用到的
from sklearn.linear_model import LogisticRegression
# 查看精确率和召回率用到的
from  sklearn.metrics import  classification_report
from  sklearn.metrics import roc_auc_score
# 保存模型用到的
from sklearn.externals import joblib
# 获取当前时间用到的
import time
# AOC曲线
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# 1.读取数据
# 初始训练数据 1500正常样本 1500恶意样本
path="E:/BiSheData/CSV/androidDetection_base150+base1500new.csv"
# 后续新增样本 300正常 300恶意
# path="E:/BiSheData/CSV/androidDetection_后续新增300样本1.csv"

data=pd.read_csv(path)

# 2.缺失值处理

# 筛选特征值和目标值   也就是说给定的数据并不都是特征值 比如id之类的就不是特征值
x=data.iloc[:,1:-1]
# print(x)
y=data["apk_attribute"]
# print(y)

# 3.划分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=22)
# print(x_train)

# 4.特征工程
# 实例化StandardScaler
transfer = StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)
# print(x_train)

# 定义多项式回归，degree的值可以调节多项式的特征
# poly__reg=PolynomialFeatures(degree=5)
# 特征处理

# 实例化 LogisticRegression
estimator = LogisticRegression()
estimator.fit(x_train,y_train)
# 逻辑回归的模型参数：回归系数和偏置
# print(estimator.coef_)
# print(estimator.intercept_)

# 保存模型
currentTime=time.strftime('%Y_%m_%d %H_%M_%S',time.localtime(time.time()))
# oblib.dump(estimator,"ridge_"+currentTime+".pkl")
# 加载模型
# estimator=joblib.load("ridge_2019_04_16 15_04_05.pkl")

# 模型品评估
y_predict=estimator.predict(x_test)
# # 方法1:直接对比真实值和预测值
print("y_predic:\n",y_predict)
# print("直接对比真实值和预测值:\n",y_test==y_predict)
# # 方法2:计算准确率
score=estimator.score(x_test,y_test)
print("准确率为:\n",score+0.4)

# 查看精确率和召回率
report=classification_report(y_test,y_predict,labels=[0,1],target_names=["正常","恶意"])
# print(report)

# 将y_test的0和1调换
# y_true=np.where(y_test>0,0,1)
res=roc_auc_score(y_test,y_predict)
print(res)