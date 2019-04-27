import numpy as np
import  matplotlib.pyplot as plt
from scipy import  stats
import seaborn as sns; sns.set()
# 随机数据
from sklearn.datasets .samples_generator import make_blobs
'''
下面那行代码中的参数含义：
n_samples:样本个数
centers：多少堆数据
random_state:随机种子，以使得每次构造的随机数据都是一样的
cluster_std:簇的离散程度,数值越大，说明数据越分散

'''
X,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.40)
print(y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plt.show()

from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(X,y)

