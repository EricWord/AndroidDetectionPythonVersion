import pandas as pd
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
import  matplotlib.pyplot as plt
# 特征工程用到的
from sklearn.preprocessing import StandardScaler

# 1.读取数据
# 初始训练数据 1500正常样本 1500恶意样本
path="E:/BiSheData/CSV/androidDetection_base150+base1500new.csv"
# 后续新增样本 300正常 300恶意
# path="E:/BiSheData/CSV/androidDetection_后续新增300样本1.csv"

data=pd.read_csv(path)

# 2.缺失值处理

# 筛选特征值和目标值   也就是说给定的数据并不都是特征值 比如id之类的就不是特征值
x=data.iloc[:,1:-1]
y=data["apk_attribute"]
transfer = StandardScaler()
x=transfer.fit_transform(x)
print(x)
# plt.scatter(x[:,0],x[:,1],c=y,s=500,cmap='autumn')
# plt.show()