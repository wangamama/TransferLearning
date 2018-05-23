import sklearn.svm
from sklearn.datasets import fetch_20newsgroups
from dataQuality.kmm import *
# ala=np.concatenate((trans_A, label_A.reshape(row_A,1)[:,-1:]), axis=1)
# s=np.concatenate((trans_S, label_S.reshape(row_S,1)[:,-1:]), axis=1)
# 初始化权重
# coef = kernel_mean_matching(s,ala,
#           kern='rbf', B=10)
# code by chenchiwei
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import tree
from scipy import sparse
from sklearn import metrics
from sklearn import svm
# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数
from KMM import kmmClassification
def tradaboost(trans_S, trans_A, label_S, label_A, test,test_label, N):
    trans_data = sparse.vstack((trans_A, trans_S))
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    # print('目标源的大小',row_S,'辅助源的大小',row_A,'测试集的大小',row_T)
    test_data = sparse.vstack((trans_data, test))


    # coef = kmmClassification.getBeta(trans_A,test.toarray(),49098)
    # weights_A = coef
    # weights_A = np.asarray(weights_A).reshape(row_A,1)
    # total=sum(weights_A[:,0])
    # for j in range(row_A):
    #     weights_A[j,0] = weights_A[j,0]/total
    # weights_S = np.ones([row_S, 1]) * np.mean(weights_A)
    weights_S = np.ones([row_S, 1])/row_S
    weights_A = np.ones([row_A, 1])/row_A
    # weights_S = np.ones([row_S, 1])
    # weights_A = np.ones([row_A, 1])
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2.0 * np.log(row_A/ N)))

    #bata = 1/(1+np.sqrt(2.0*np.log(row_A)/N));

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])

    # trans_data = np.asarray(trans_data, order='C')
    # trans_label = np.asarray(trans_label, order='C')
    # test_data = np.asarray(test_data, order='C')

    # print(trans_data.shape)
    # print(test_data.shape)
    accuracy_scorelist=[]
    f1_scorelist=[]
    recall_scorelist=[]
    for i in range(N):
        P = calculate_P(weights, trans_label)

        result_label[:, i] = train_classify(trans_data, trans_label,
                                            test_data, P)

        error_rate = 0.0
        for j in range(row_A, row_A + row_S):
            error_rate += (weights[j] * abs(result_label[j, i] - trans_label[j]))
        error_rate = error_rate / sum(weights[row_A:])


        #error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
        #                                  weights[row_A:row_A + row_S, :])
        #print ('Error rate:', error_rate)
        # if error_rate != 1:
        #     bata_T[0, i] = error_rate / (1.0 - error_rate)
        # if error_rate >= 0.5 and error_rate != 1:
        #     bata_T[0, i] = 0.45 / (0.51)
        # if error_rate == 1:
        #     bata_T[0, i] = 0.4

        if error_rate >= 0.5:
            #error_rate = 0.5
            error_rate = 0.499;
        if error_rate == 0:
            #error_rate = 0.000001
            #error_rate=0.0001
            error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)
        # Ct = 2 * (1 - error_rate);
        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_A + j, i] - trans_label[row_A+j])))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - trans_label[j]))


            ##每次迭代完成计算下在测试集合上的误差
        # predic_temp = np.zeros([row_T])
        # iteration = i;
        # for i in range(row_T):
        #     left = np.sum(
        #         result_label[row_A + row_S + i, int(np.ceil(iteration / 2)):iteration] * np.log(1 / bata_T[0, int(np.ceil(iteration / 2)):iteration]))
        #     right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(iteration / 2)):iteration]))
        #     if left >= right:
        #         predic_temp[i] = 1
        #     else:
        #         predic_temp[i] = 0
        # accuracy_scorelist.append(metrics.accuracy_score(test_label, predic_temp))
        # recall_scorelist.append(metrics.recall_score(test_label, predic_temp))
        # f1_scorelist.append(metrics.f1_score(test_label, predic_temp))
    # print bata_T
    for i in range(row_T):
        # 跳过训练数据的标签
        # left = np.sum(
        #     result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        # right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        left = np.sum(
            result_label[row_A + row_S + i, 0:N] * np.log(1 / bata_T[0, 0:N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, 0:N]))
        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]
        # predict[i] = left - right;
    return predict, accuracy_scorelist, recall_scorelist, f1_scorelist


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights)/total

from sklearn.linear_model import LogisticRegression

def train_classify(trans_data, trans_label, test_data, P):
    clf = LogisticRegression()
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)


# def calculate_error_rate(label_R, label_H, weight):
#     total = np.sum(weight)
#     #return np.sum((weight[:, 0] / total)* np.abs(label_R - label_H))
#     return  return_correct_rate(label_R,label_H)
