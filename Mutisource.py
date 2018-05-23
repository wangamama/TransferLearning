# code by chenchiwei
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import tree
from sklearn import svm
import math
from sklearn.tree import DecisionTreeRegressor

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    return out


# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数

# source_sum 源领域的数目
# soucenum   源领域所有样本的数目

def Mutisource_tradaboost(trans_S, trans_A_list, label_S, label_A_list, test, N,source_sum,soucenum):


    #首先计算bata
    bata = 1 / (1 + np.sqrt(2 * np.log(soucenum) / N))
    row_S = trans_S.shape[0]
    row_T = test.shape[0]
    weights_S=  np.ones([row_S, 1])/row_S
    weights_A=[]
    train=[]
    train_lable=[]
    test_data=[]
    result_label=[]
    # 存储每次迭代的标签和bata值
    bata_T = np.zeros([1, N])
    result_labelsum = np.zeros([row_S + row_T, N])
    for i in range(source_sum):
        row_A = trans_A_list[i].shape[0]
        weights_A.append(np.ones([row_A, 1])/row_A)
        train.append(np.concatenate((trans_A_list[i], trans_S), axis=0))
        train_lable.append(np.concatenate((label_A_list[i], label_S), axis=0))
        test_data.append( np.concatenate((train[i], test), axis=0))
        result_label.append(np.ones([row_A + row_S + row_T, N]))
    #生成初始的权重
    for i in range(N):
        #将权重向量归一化
        error_list=[]
        max_error_list=[]
        for j in range(source_sum):
            row_A = trans_A_list[j].shape[0]
            weights = np.concatenate((weights_A[j], weights_S), axis=0)
            P = calculate_P(weights)
            result_label[j][:, i] = train_classify(train[j], train_lable[j],
                                                test_data[j], P)
            temp = np.abs(result_label[j][row_A:row_A + row_S, i] - train_lable[j][row_A:])
            #temp = np.abs(result_label[j][:row_A + row_S, i] - train_lable[j])
            error_max = temp.max()
            max_error_list.append(error_max)
            error_rate = 0.0
            for m in range(row_A, row_A + row_S):
                error_rate += (weights_S[m-row_A] * ((abs(result_label[j][m, i] - train_lable[j][m]) / error_max)))
            error_rate = error_rate / sum(weights_S)
            error_list.append(error_rate)
        g=[]
        #g=error_list;
        for j in range(source_sum):
            g.append(math.exp(1-error_list[j])/math.exp(error_list[j]))
        g = [x/sum(g) for x in g]

        for j in range(source_sum):
            row_A = trans_A_list[j].shape[0]
            result_labelsum[:, i]=result_labelsum[:, i]+g[j]*result_label[j][row_A:row_A + row_S+row_T, i]

        temp = np.abs(result_labelsum[:row_S, i] - label_S)
        error_max = temp.max()
        error_rate = 0.0
        for m in range(row_S):
            error_rate += (weights_S[m] * ((abs(result_labelsum[m, i] - label_S[m]) / error_max)))
        error_rate = error_rate / sum(weights_S)


        #更新样本权重
        if error_rate >= 0.5:
            error_rate = 0.499;
        if error_rate == 0:
            error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)
        for j in range(row_S):
            weights_S[j] = weights_S[j] * np.power(bata_T[0, i], 1-((abs(result_labelsum[j, i] - label_S[j]) /error_max)))
        for j in range(source_sum):
            row_A = trans_A_list[j].shape[0]
            temp = np.abs(result_label[j][:row_A, i] - label_A_list[j])
            error_max = temp.max()
            for m in range(row_A):
                if (abs(result_label[j][m, i] - label_A_list[j][m]) > 0):
                    weights_A[j][m]=weights_A[j][m] * np.power(bata, abs(result_label[j][m, i] - label_A_list[j][m]) / error_max)

        # bata_T[0, i] = (1/2)*math.log((1 - error_rate)/error_rate)
        # for j in range(row_S):
        #     weights_S[j] = weights_S[j] * np.exp(bata_T[0, i]*((abs(result_labelsum[j, i] - label_S[j]) / error_max)))
        # for j in range(source_sum):
        #     row_A = trans_A_list[j].shape[0]
        #     temp = np.abs(result_label[j][:row_A, i] - label_A_list[j])
        #     error_max = temp.max()
        #     for m in range(row_A):
        #         if (abs(result_label[j][m, i] - label_A_list[j][m]) > 0.04):
        #             weights_A[j][m] = weights_A[j][m] * np.exp(-bata*abs(
        #                 result_label[j][m, i] - label_A_list[j][m]) / error_max)

    #
    # predictions=result_labelsum[row_S:,0:N]
    # # Sort the predictions
    # sorted_idx = np.argsort(predictions, axis=1)
    # # Find index of median prediction for each sample
    # #bata_T = 1/bata_T[0, 0:N]
    # bata_T = np.log(1/bata_T[0, :N])
    # bata_T[:] = bata_T[:] / np.sum(bata_T[:])
    # weight_cdf = stable_cumsum(bata_T[sorted_idx], axis=1)
    # median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    # median_idx = median_or_above.argmax(axis=1)
    # median_estimators = sorted_idx[np.arange(test.shape[0]), median_idx]
    # # Return median predictions
    # return predictions[np.arange(test.shape[0]), median_estimators]


    predictions = result_labelsum[row_S:, int(np.ceil(N / 2)):N]
    # Sort the predictions
    sorted_idx = np.argsort(predictions, axis=1)
    # Find index of median prediction for each sample
    bata_T = np.log(1 / bata_T[0, int(np.ceil(N / 2)):N])
    #bata_T =  bata_T[0, int(np.ceil(N / 2)):N]
    bata_T[:] = bata_T[:] / np.sum(bata_T[:])
    weight_cdf = stable_cumsum(bata_T[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    median_idx = median_or_above.argmax(axis=1)
    median_estimators = sorted_idx[np.arange(test.shape[0]), median_idx]
    # Return median predictions
    return predictions[np.arange(test.shape[0]), median_estimators]



def calculate_P(weights):
    total = np.sum(weights)
    return weights/total

from sklearn import neighbors
def train_classify(trans_data, trans_label, test_data, P):
    clf = DecisionTreeRegressor(max_depth=3)
    #clf = neighbors.KNeighborsRegressor()
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