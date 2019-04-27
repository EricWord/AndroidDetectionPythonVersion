import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from  sklearn.datasets import  make_gaussian_quantiles
from sklearn.preprocessing import PolynomialFeatures
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本，2个样本特征
# path="E:/BiSheData/CSV/androidDetection_all.csv"
path="E:/BiSheData/CSV/androidDetection_后续新增300样本1.csv"

data=pd.read_csv(path)
x_data=data.iloc[:,1:-1]
# print(x)
y_data=data["apk_attribute"]
# 可以生成两类或者多类数据
# x_data,y_data=make_gaussian_quantiles(n_samples=500,n_features=2)
# plt.scatter(x_data[:,0],x_data[:,1],x_data[:,2],c=y_data)
# plt.show()

logistic=linear_model.LogisticRegression()
logistic.fit(x_data,y_data)
#
# # 定义多项式回归，degree的值可以调节多项式的特征
poly_reg=PolynomialFeatures(degree=3)
# # 特征处理
x_poly=poly_reg.fit_transform(x_data)
# # 定义逻辑回归模型
logistic.fit(x_poly,y_data)
#
# # 获取数据值所在的范围
# x_min,x_max=x_data[:,1].min()-1,x_data[:,1].max()+1
# y_min,y_max=x_data[:,-1].min()-1,x_data[:,-1].max()+1
#
# # 生成网格矩阵
# xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
#
# z=logistic.predict(poly_reg.fit_transform(np.c_[xx.ravel(),yy.ravel()]))
# z=z.reshape(xx.shape)
#
# # 登高线图
# cs=plt.contourf(xx,yy,z)
# # 样本散点图
# plt.scatter(x_data[:,0],x_data[:,1],c=y_data)
# plt.show()
print("预测准确率：",logistic.score(x_poly,y_data))



