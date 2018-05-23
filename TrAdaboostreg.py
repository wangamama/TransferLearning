# code by chenchiwei
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import tree
from sklearn import svm
import math
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    return out


# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数
def tradaboost(trans_S, trans_A, label_S, label_A, test, y_target_test,N,islog):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_data, test), axis=0)

    # 初始化权重
    weights_A = np.ones([row_A, 1])/row_A
    weights_S = np.ones([row_S, 1])/row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A/N)))

    # 存储每次迭代的标签和bata值
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])

    #print ('params initial finished.')


    for i in range(N):
        #将权重向量归一化
        P = calculate_P(weights)


        result_label[:, i] = train_classify(trans_data, trans_label,
                                            test_data, weights,test,y_target_test,islog)

        temp0 = np.abs(result_label[:row_A + row_S, i] - trans_label)
        error_max0 = temp0.max()
        temp = np.abs(result_label[row_A:row_A + row_S, i] - label_S)
        error_max = temp.max()
        if error_max0==0.0 or error_max==0.0:
            N=i;
            break
        temp2 = np.abs(result_label[:row_A, i] - label_A)
        error_max2 = temp2.max()
        error_rate = 0.0
        for j in range(row_A, row_A + row_S):
            error_rate += (weights[j] * ((abs(result_label[j, i] - trans_label[j])/error_max0)))
        error_rate = error_rate / sum(weights[row_A:])
        if error_rate >= 0.5:
            error_rate = 0.499;
        if error_rate == 0:
            error_rate=0.001
        bata_T[0, i] = error_rate / (1 - error_rate)

        for j in range(row_S):
                weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], -(
                np.abs(result_label[row_A + j, i] - label_S[j]) / error_max0))
        # 调整辅域样本权重
        for j in range(row_A):
            if islog:
                if (abs(result_label[j, i] - label_A[j]) >0):#0.02872
                        weights[j] = weights[j] * np.power(bata, np.abs((result_label[j, i] - label_A[j])/error_max0))
            else:
                 weights[j] = weights[j] * np.power(bata, np.abs((result_label[j, i] - label_A[j]) / error_max0))
    # bata_T[0,:]=bata_T[0,:]/np.sum(bata_T[0,:])

    #
    predictions=result_label[row_A + row_S:,int(np.ceil(N / 2)):N]
    # Sort the predictions
    sorted_idx = np.argsort(predictions, axis=1)
    # Find index of median prediction for each sample
    bata_T = np.log(1/bata_T[0, int(np.ceil(N / 2)):N])
    #bata_T =  bata_T[0, int(np.ceil(N / 2)):N]
    bata_T[:] = bata_T[:] / np.sum(bata_T[:])
    weight_cdf = stable_cumsum(bata_T[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    median_idx = median_or_above.argmax(axis=1)
    median_estimators = sorted_idx[np.arange(test.shape[0]), median_idx]
    # Return median predictions
    return predictions[np.arange(test.shape[0]), median_estimators]
    #
    # for i in range(row_T):
    #     # 跳过训练数据的标签
    #     # predict[i]=np.median(
    #     #     result_label[row_A + row_S + i, :] * np.log(1 / bata_T[0, :]))
    #     # predict[i] = np.sum(
    #     #     result_label[row_A + row_S + i, :] * (1-bata_T[0, :]))
    #
    #     predict[i] = weighted_median(result_label[row_A + row_S + i,int(np.ceil(N / 2)):N],
    #                                  np.log(1 / bata_T[0,int(np.ceil(N / 2)):N]))
    # return predict

def calculate_P(weights):
    total = np.sum(weights)
    return weights/total


def train_classify(trans_data, trans_label, test_data, P,test,y_target_test,islog,):
    # if islog:
    #     clf = svm.SVR(C=100)
    # else:
    clf = DecisionTreeRegressor(max_depth=3)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)

def weighted_median(values, weights):
    ''' compute the weighted median of values list. The
weighted median is computed as follows:
    1- sort both lists (values and weights) based on values.
    2- select the 0.5 point from the weights and return the corresponding values as results
    e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are probabilities.
    sorted values = [0, 1, 3] and corresponding sorted weights = [0.6,     0.1, 0.3] the 0.5 point on
    weight corresponds to the first item which is 0. so the weighted     median is 0.'''

    #convert the weights into probabilities
    sum_weights = sum(weights)
    weights = np.array([(w*1.0)/sum_weights for w in weights])
    #sort values and weights based on values
    values = np.array(values)
    sorted_indices = np.argsort(values)
    values_sorted  = values[sorted_indices]
    weights_sorted = weights[sorted_indices]
    #select the median point
    it = np.nditer(weights_sorted, flags=['f_index'])
    accumulative_probability = 0
    median_index = -1
    while not it.finished:
        accumulative_probability += it[0]
        if accumulative_probability > 0.5:
            median_index = it.index
            return values_sorted[median_index]
        elif accumulative_probability == 0.5:
            median_index = it.index
            it.iternext()
            next_median_index = it.index
            return np.mean(values_sorted[[median_index, next_median_index]])
        it.iternext()

    return values_sorted[median_index]

from sklearn.ensemble import AdaBoostRegressor