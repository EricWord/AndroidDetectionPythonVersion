import pandas as pd
# 特征工程用到的
from sklearn.preprocessing import  StandardScaler
from sklearn.externals import joblib
# 控制台调用Java程序传递参数用到
import sys

'''
CSV文件存放路径
模型存放路径
'''
def predict(csvPath,modelPath):
    # 读取数据
    data = pd.read_csv(csvPath)
    # 特征
    x = data.iloc[:, 1:-1]
    # 特征工程
    # 实例化StandardScaler
    transfer = StandardScaler()
    x = transfer.fit_transform(x)
    # 加载模型
    estimator = joblib.load(modelPath)
    # 调用模型进行预测
    predictResult=estimator.predict(x)
    # 返回预测结果
    res=""
    if(predictResult==0):
        res="正常"
    else:
        res="恶意"

    return res

if __name__ == "__main__":
    # print(predict("C:\\AndroidDetection\\temp\\srcApkFeature.csv","C:\\AndroidDetection\\temp\\predict_model.pkl"))
    # print(predict("E:\\BiSheData\\CSV\\test.csv","E:\\BiSheData\\temp\\predict_model.pkl"))
    a = []
    for i in range(0, len(sys.argv)):
        a.append((sys.argv[i]))

    print("第一个参数为:"+a[0], ",第二个参数为:"+a[1])
    # print(predict(a[0], a[1]))