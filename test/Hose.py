from numpy import *
import numpy as np


def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('E:\BiSheData\CSV\\androidDetection_base150+base1500new.csv')
    for line in fr.readlines():
        lineArr = line.strip().split(",")
        # print(int(lineArr[2]))
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        for ele in lineArr:
            dataMat.append(ele)
            # 第一列是1，另外加的，即形如[1,x1,x2]
        labelMat.append(lineArr[136])
        # print(labelMat)
    return dataMat, labelMat  # 此时是数组


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))  # 返回一个0-1的100*1的数组


# 梯度上升
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix  # 转换成numpy矩阵(matrix),dataMatrix:100*3
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix # transpose转置为100*1
    m, n = shape(dataMatrix)  # 得到行和列数，0代表行数，1代表列数
    alpha = 0.001  # 步长
    maxCycles = 500  # 最大迭代次数
    weights = ones((n, 1))  # 3*1的矩阵，其权系数个数由给定训练集的维度值和类别值决定
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix mult # 利用对数几率回归，z = w0 +w1*X1+w2*X2，w0相当于b
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult  # 一种极大似然估计的推到过程，但每次的权系数更新都要遍历整个数据集
    return weights


# 画出数据集和Logistic分类直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()  # 加载数据集，训练数据和标签
    dataArr = array(dataMat)  # 进行数据处理，必须转换为numpy中的array
    n = shape(dataArr)[0]  # 样本个数
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])  # 一类
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])  # 另一类
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # 散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # print(type(x))
    y = (-weights[0] - weights[1] * x) / weights[2]  # 换算后的简化形式
    y1 = y.T  # 转置换类型 或者其他api都可以试试

    # print(type(y))
    # m,b = np.polyfit(x,y)
    ax.plot(x, y1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 梯度上升算法
"""
设定循环迭代次数，权重系数的每次更新是荣国计算所有样本得出来的，当训练集过于庞大不利于计算
"""


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01  # x学习率
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 随机梯度上升
"""
对梯度上升算法进行改进，权重系数的每次更新通过训练集的每个记录计算得到
可以在新样本到来时分类器进行增量式更新，因为随机梯度算法是个更新权重的算法
但是算法容易受到噪声点的影响。在大的波动停止后，还有小周期的波动
"""

"""
前者的h,error 都是数值，前者没有矩阵转换过程，后者都是向量，数据类型则变为numpy
"""


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            # 学习率：变化。随着迭代次数的增多，逐渐变下。
            # 权重系数更新：设定迭代次数，每次更新选择的样例是随机的（不是依次选的）。
            weights = weights + alpha * error * dataMatrix[randIndex]  # 分类函数
            # 为什么删除变量？
            # print(list(dataIndex)[randIndex])
            del (list(dataIndex)[randIndex])
    return weights


# 分类算法
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 测试算法
"""
为了量化回归效果，使用错误率作为观察指标。根据错误率决定是否回退到训练阶段，通过改变迭代次数和步骤等参数来得到更好的回归系数。
"""


def colicTest():
    frTrain = open('E:\\BiSheData\\CSV\\androidDetection_all.csv', encoding='utf8');
    frTest = open('E:\\BiSheData\\CSV\\androidDetection_后续新增300样本1.csv', encoding='utf8')
    trainingSet = []
    trainingLabels = []
    # print(frTest)
    # print(frTrain)
    for line in frTrain.readlines():
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(135):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


# 通过多次测试，取平均，作为该分类器错误率

"""
优点：计算代价不高，易于理解和实现；

缺点：容易欠拟合，分类进度可能不高；

使用数据类型：数值型和标称型数据。

"""


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    # 入口函数:模型参数y=wx
    dataArr, labelMat = loadDataSet()
    # weights = gradAscent(dataArr, labelMat)  # dataArr:100*3,labelMat是一个列表
    weights = stocGradAscent1(array(dataArr), labelMat, 150)
    plotBestFit(weights)
    multiTest()
