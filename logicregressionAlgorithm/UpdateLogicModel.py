import pandas as pd
# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# 特征工程用到的
from sklearn.preprocessing import  StandardScaler

# 保存模型用到的
from sklearn.externals import joblib
# 控制台调用Java程序传递参数用到
import sys

def updateModel(csvPath,preModelPath,newModelSavePath):
    # 读取数据
    data=pd.read_csv(csvPath)
    # 特征
    x=data.iloc[:,1:-1]
    # 标签
    y=data["apk_attribute"]

    # 特征工程
    # 实例化StandardScaler
    transfer = StandardScaler()
    x_train=transfer.fit_transform(x)
    # 加载模型
    estimator = joblib.load(preModelPath)
    estimator.fit(x_train,y)



    # 保存模型 保存路径
    # "E:\\BiSheData\\temp\\predict_model_after_update.pkl"
    joblib.dump(estimator,newModelSavePath)


    return  "模型更新完成！"


if __name__ == "__main__":
    a = []
    for i in range(1, len(sys.argv)):
        a.append((sys.argv[i]))

    print(updateModel(a[0],a[1],a[2]))