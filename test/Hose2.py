# ------------------------------实例：从疝气病预测病马的死亡率----------------------------
# 1 准备数据：处理数据的缺失值
# 这里将特征的缺失值补0，从而在更新时不影响系数的值
# 预处理数据
from math import exp

from numpy import mat, shape, array, ones
from pandas.tests.arithmetic.conftest import one
from sklearn.utils import random


def loadDataSet():
    # 创建两个列表
    dataMat = [];
    labelMat = []
    # 打开文本数据集
    fr = open('testSet.txt')
    # 遍历文本的每一行
    for line in fr.readlines():
        # 对当前行去除首尾空格，并按空格进行分离
        lineArr = line.strip().split(',')
        # 将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 将当前行标签添加到标签列表
        labelMat.append(int(lineArr[2]))
    # 返回数据列表，标签列表
    return dataMat, labelMat


# 定义sigmoid函数
def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))


# 梯度上升法更新最优拟合参数
# @dataMatIn：数据集
# @classLabels：数据标签
def gradAscent(dataMatIn, classLabels):
    # 将数据集列表转为Numpy矩阵
    dataMatrix = mat(dataMatIn)
    # 将数据集标签列表转为Numpy矩阵，并转置
    labelMat = mat(classLabels).transpose()
    # 获取数据集矩阵的行数和列数
    m, n = shape(dataMatrix)
    # 学习步长
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 初始化权值参数向量每个维度均为1.0
    weights = one((n, 1))
    # 循环迭代次数
    for k in range(maxCycles):
        # 求当前的sigmoid函数预测概率
        h = sigmoid(dataMatrix * weights)
        # ***********************************************
        # 此处计算真实类别和预测类别的差值
        # 对logistic回归函数的对数释然函数的参数项求偏导
        error = (labelMat - h)
        # 更新权值参数
        weights = weights + alpha * dataMatrix.transpose() * error
        # ***********************************************
    return weights


# 2 分类决策函数
def clasifyVector(inX, weights):
    # 计算logistic回归预测概率
    prob = sigmoid(inX * weights)
    # 大于0.5预测为1
    if prob > 0.5:
        return 1.0
    # 否则预测为0
    else:
        return 0.0


# @dataMatrix：数据集列表
# @classLabels：标签列表
# @numIter：迭代次数，默认150
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 将数据集列表转为Numpy数组
    dataMat = array(dataMatrix)
    # 获取数据集的行数和列数
    m, n = shape(dataMat)
    # 初始化权值参数向量每个维度均为1
    weights = ones(n)
    # 循环每次迭代次数
    for j in range(numIter):
        # 获取数据集行下标列表
        dataIndex = range(m)
        # 遍历行列表
        for i in range(m):
            # 每次更新参数时设置动态的步长，且为保证多次迭代后对新数据仍然具有一定影响
            # 添加了固定步长0.1
            alpha = 4 / (1.0 + j + i) + 0.1
            # 随机获取样本
            randomIndex = int(random.nuiform(0, len(dataIndex)))
            # 计算当前sigmoid函数值
            h = sigmoid(dataMat[randomIndex] * weights)
            # 计算权值更新
            # ***********************************************
            error = classLabels - h
            weights = weights + alpha * error * dataMat[randomIndex]
            # ***********************************************
            # 选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
            del (dataIndex[randomIndex])
    return weights


# logistic回归预测算法
def colicTest():
    # 打开训练数据集
    frTrain = open('horseColicTraining.txt')
    # 打开测试数据集
    frTest = open('horseColicTest.txt')
    # 新建两个孔列表，用于保存训练数据集和标签
    trainingSet = []
    trainingLabels = []
    # 读取训练集文档的每一行
    for line in frTrain.readlines():
        # 对当前行进行特征分割
        currLine = line.strip().split()
        # 新建列表存储每个样本的特征向量
        lineArr = []
        # 遍历每个样本的特征
        for i in range(21):
            # 将该样本的特征存入lineArr列表
            lineArr.append(float(currLine[i]))
        # 将该样本标签存入标签列表
        trainingLabels.append(currLine[21])
        # 将该样本的特征向量添加到数据集列表
        trainingSet.append(lineArr)
    # 调用随机梯度上升法更新logistic回归的权值参数
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    # 统计测试数据集预测错误样本数量和样本总数
    errorCount = 0;
    numTestVec = 0.0
    # 遍历测试数据集的每个样本
    for line in frTest.readlines():
        # 样本总数加1
        numTestVec += 1.0
        # 对当前行进行处理，分割出各个特征及样本标签
        currLine = line.strip().split()
        # 新建特征向量
        lineArr = []
        # 将各个特征构成特征向量
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 利用分类预测函数对该样本进行预测，并与样本标签进行比较
        if (clasifyVector(lineArr, trainWeights) != currLine[21]):
            # 如果预测错误，错误数加1
            errorCount += 1
    # 计算测试集总的预测错误率
    errorRate = (float(errorCount) / numTestVec)
    # 打印错误率大小
    print('the error rate of this test is: %f', (errorRate))
    # 返回错误率
    return errorRate


# 多次测试算法求取预测误差平均值
def multTest():
    # 设置测试次数为10次，并统计错误率总和
    numTests = 10;
    errorRateSum = 0.0
    # 每一次测试算法并统计错误率
    for k in range(numTests):
        errorRateSum += colicTest()
    # 打印出测试10次预测错误率平均值
    print('after %d iterations the average error rate is: %f', (numTests, errorRateSum / float(numTests)))
