import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_gaussian_quantiles
import sklearn.svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from scipy import interp
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import TR.TrAdaboostkmm as kmmtr
import TR.TrAdaboost2 as tr
import TR.SPY as SPY


import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
from  text import util
from sklearn.metrics import roc_curve, auc

from TR.classification_report import  *
import time


def  main():



# '''
#     定义一个二分类问题，分类rec和sci，但是目标领域和源领域的数据来源不同
#     如何控制源领域和目标领域数据的数量
# '''


    # target_categories = ["rec.sport.hockey", "rec.motorcycles","sci.crypt", "sci.electronics"]
    # source_categories = ["rec.sport.baseball", "rec.autos","sci.med", "sci.space"]


    target_categories = ["rec.autos", "rec.sport.baseball","sci.med", "sci.space"]
    source_categories = ["rec.motorcycles", "rec.sport.hockey","sci.crypt", "sci.electronics"]

    #实验组1
    target_categories = ["rec.autos", "sci.med"]
    source_categories = ["rec.sport.hockey", "sci.electronics"]

    #实验组2
    target_categories = ["comp.graphics", "rec.autos"]
    source_categories = ["comp.os.ms-windows.misc", "rec.sport.hockey"]

    # target_categories = ["rec.sport.hockey", "rec.motorcycles"]
    # source_categories = ["sci.med", "sci.space"]

    target_categories = ["sci.crypt", "sci.space", "talk.politics.guns", "talk.politics.mideast"]
    source_categories = ["sci.electronics", "sci.med", "talk.politics.misc", "talk.religion.misc"]


    # 实验组1
    target_categories = ["rec.autos", "sci.med"]
    source_categories = ["rec.sport.hockey", "sci.electronics"]

    target = fetch_20newsgroups(subset='test',categories = target_categories, shuffle = True, random_state = 42)
    source= fetch_20newsgroups(subset='test',categories = source_categories, shuffle = True, random_state = 42)

    # source.data = source.data[0:1000]
    # source.target = source.target[0:1000]
    #
    target.data = target.data[0:400]
    target.target = target.target[0:400]

    print(target.target)
    print(target.target_names)
    print(source.target_names)

    print('目标源的大小', len(target.data), '辅助源的大小', len(source.data))

    # #
    # target.target[target.target == 0] = 0
    # target.target[target.target == 1] = 0
    # target.target[target.target == 2] = 1
    # target.target[target.target == 3] = 1
    # # print(type(target.target))
    # # print(target.target)
    #
    # source.target[source.target == 0] = 0
    # source.target[source.target == 1] = 0
    # source.target[source.target == 2] = 1
    # source.target[source.target == 3] = 1


    merge_target_source = np.concatenate((target.data, source.data), axis=0)
    merge_target_source_label = np.concatenate((target.target, source.target), axis=0)
    # print(set(merge_target_source_label))

    # refine emails - delete unwanted text form them
    util.refine_all_emails(merge_target_source)
    # feature Extractoin
    # BOW Bag Of Words
    TFIDF = util.bagOfWords(merge_target_source)
    #TFIDF = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(TFIDF)
    #TFIDF = sklearn.feature_extraction.text.TfidfTransformer.transform(TFIDF)

    length=len(target.data)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(TFIDF[:length], merge_target_source_label[:length], test_size=0.6,random_state = 0)

    #X_train_temp, X_test, y_train_temp, y_test=sklearn.cross_validation.train_test_split(X_test,y_test)

    print("测试集的大小",y_test.shape)
    TFIDF = np.array(TFIDF.toarray())
    merge_target_source_label=np.array(merge_target_source_label)
    print((X_train.shape))
    print((TFIDF.shape))
    # build classifier
    # clf = sklearn.svm.LinearSVC()


    clf = LogisticRegression()


    # print("辅助数据集和目标数据集一起训练",split_test_classifier(clf, X,
    #       np.concatenate((y_train[:,None], merge_target_source_label[length:,None]), axis=0)
    #                             ,X_test[0:200,:],y_test[0:200,None]))
    X=sparse.vstack((X_train[:,:], TFIDF[length:,:]))
    print("辅助数据集和目标数据集一起训练", split_test_classifier(clf, X,
                                                   np.concatenate(
                                                       (y_train[:, None], merge_target_source_label[length:, None]), axis=0)
                                                   , X_test, y_test))
    # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(TFIDF, merge_target_source_label, test_size=0.4,random_state = 0)


    clf = LogisticRegression()
    # print("目标数据集单独训练",split_test_classifier(clf,X_train, y_train
    #                             , X_test[0:200,:], y_test[0:200,None]))
    print("目标数据集单独训练",split_test_classifier(clf,X_train, y_train
                            , X_test, y_test))
    #def fit(self, diff_train, diff_label, same_train, same_train_label, MAX_ITERATION=30):
    #model.fit(TFIDF[length:,:], merge_target_source_label[length:,None], X_train, y_train[:,None],MAX_ITERATION=100)

    #
    # predict1, accuracy_scorelist1, recall_scorelist1, f1_scorelist1 = SPY.tradaboost(X_train, TFIDF[length:, :],
    #                                                                                    y_train[:, None],
    #                                                                                    merge_target_source_label[
    #                                                                                    length:, None],
    #                                                                                    X_test, y_test, 55, True)
    # print_classification_report('LR-TRadaboost', predict1, y_test)

    start =time.clock()
    predict2, accuracy_scorelist2, recall_scorelist2, f1_scorelist2 = tr.tradaboost(
        X_train, TFIDF[length:, :], y_train[:, None], merge_target_source_label[length:, None], X_test, y_test, 85)
    print_classification_report('TRadaboost', predict2, y_test)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))

    # 原生的tradaboost
    Predict = [];

    reslist = [];
    reslistx = [];
    # for i in range(5, 200, 10):
    #     predict2, accuracy_scorelist2, recall_scorelist2, f1_scorelist2 = tr.tradaboost(
    #         X_train, TFIDF[length:, :], y_train[:, None], merge_target_source_label[length:, None], X_test, y_test, i)
    #     reslist.append(return_correct_rate(predict2, y_test))
    #     reslistx.append(i)
        #Predict = predict2

    #
    # reslist=[0.775, 0.825, 0.875, 0.875, 0.8875, 0.8958333333333334, 0.8958333333333334, 0.8791666666666667, 0.8875,
    #  0.8833333333333333, 0.8833333333333333, 0.8916666666666667, 0.9166666666666666, 0.9208333333333333, 0.9125,
    #  0.9208333333333333, 0.9125, 0.9125, 0.9041666666666667, 0.9]

    #print_classification_report('TRadaboost', Predict, y_test)
    print(reslist)


    # plt.plot(reslistx, reslist, marker='+', linestyle='dashed', linewidth=1,label="tradaboost")  # plt.plot(range(5,31,5), accuracy_scorelist[4:30:5],marker='x', linestyle='dashed',linewidth=1,label="vfkmm without eliminate")
    # plt.xlabel("迭代次数")
    # plt.ylabel("score")
    # plt.legend(loc="lower right")
    # plt.show()


    start = time.clock()
    #kmm排除低权重的样本
    predict1, accuracy_scorelist1, recall_scorelist1, f1_scorelist1 = kmmtr.tradaboost(X_train, TFIDF[length:, :],
                                                                                   y_train[:, None],
                                                                                   merge_target_source_label[length:, None],
                                                                                   X_test, y_test,20, True)
    #
    #
    print_classification_report('KMM-TRadaboost',predict1,y_test)


    end = time.clock()
    print('Running time: %s Seconds' % (end - start))

    SPY
    reslisttempx = []
    reslisttemp = [];
    for i in range(5, 80, 2):
        # predict3, accuracy_scorelist, recall_scorelist, f1_scorelist = SPY.tradaboost(
        #     X_train, TFIDF[length:, :], y_train[:, None], merge_target_source_label[length:, None], X_test, y_test, i,
        #     True)
        # reslisttemp.append(return_correct_rate(predict3, y_test))
        reslisttempx.append(i)
        #Predict = predict3
    #print_classification_report('lr-TRadaboost', Predict, y_test)
    reslisttemp=[0.875, 0.8791666666666667, 0.8958333333333334, 0.9083333333333333, 0.9041666666666667, 0.8916666666666667, 0.9, 0.9, 0.9041666666666667, 0.9083333333333333, 0.9, 0.9, 0.9083333333333333, 0.9041666666666667, 0.9041666666666667, 0.9083333333333333, 0.9041666666666667, 0.9041666666666667, 0.9041666666666667, 0.9041666666666667, 0.9041666666666667, 0.9125, 0.9208333333333333, 0.9291666666666667, 0.9291666666666667, 0.9291666666666667, 0.9291666666666667, 0.9291666666666667, 0.9291666666666667, 0.9291666666666667, 0.925, 0.925, 0.9208333333333333, 0.9208333333333333, 0.9166666666666666, 0.9166666666666666, 0.9083333333333333, 0.9041666666666667]

    print(reslisttemp)
    print(reslisttempx)







    reslisttempx1 = []
    reslisttemp1 = [];
    for i in range(5, 80, 2):
        predict1, accuracy_scorelist1, recall_scorelist1, f1_scorelist1 = kmmtr.tradaboost(X_train, TFIDF[length:, :],
                                                                                           y_train[:, None],
                                                                                           merge_target_source_label[
                                                                                           length:, None],
                                                                                           X_test, y_test, i, True)
        reslisttemp1.append(return_correct_rate(predict1, y_test))
        reslisttempx1.append(i)
        Predict = predict1
    print_classification_report('KMM-TRadaboost', Predict, y_test)
    print(reslisttemp1)
    print(reslisttempx1)

    plt.plot(reslisttempx1, reslisttemp1, marker='o', linestyle='dashed', linewidth=1, label="vfkmm tradaboost")
    plt.plot(reslistx, reslist, marker='+', linestyle='dashed', linewidth=1,label="tradaboost")  # plt.plot(range(5,31,5), accuracy_scorelist[4:30:5],marker='x', linestyle='dashed',linewidth=1,label="vfkmm without eliminate")
    plt.plot(reslisttempx, reslisttemp, marker='x', linestyle='dashed', linewidth=1, label="LR tradaboost")
    plt.xlabel("迭代次数")
    plt.ylabel("score")
    plt.legend(loc="lower right")
    plt.show()



    # # SPY
    # reslisttempx=[]
    # reslisttemp = [];
    # for i in range(5,100,2):
    #     predict3, accuracy_scorelist, recall_scorelist, f1_scorelist = SPY.tradaboost(
    #         X_train, TFIDF[length:, :], y_train[:, None], merge_target_source_label[length:, None], X_test, y_test, i,
    #         True)
    #     reslisttemp.append(return_correct_rate(predict3,y_test))
    #     reslisttempx.append(i)
    #     Predict = predict3
    # print_classification_report('lr-TRadaboost', Predict, y_test)
    # print(reslisttemp)

    # predict, accuracy_scorelist, recall_scorelist, f1_scorelist =SPY.tradaboost(
    #     X_train, TFIDF[length:, :], y_train[:, None],merge_target_source_label[length:, None],X_test,y_test, 62,True)
    # print_classification_report('SPY',predict,y_test)
    #
    # predict2, accuracy_scorelist2, recall_scorelist2, f1_scorelist2=tr.tradaboost(
    #     X_train, TFIDF[length:, :], y_train[:, None],merge_target_source_label[length:, None],X_test,y_test, 100)
    # print_classification_report('TRadaboost',predict2,y_test)







    # # kmm排除低权重的样本
    # predict1, accuracy_scorelist1, recall_scorelist1, f1_scorelist1 = kmmtr.tradaboost(X_train, TFIDF[length:, :],
    #                                                                                y_train[:, None],
    #                                                                                merge_target_source_label[length:, None],
    #                                                                                X_test, y_test,65, True)
    # #
    # #
    # print_classification_report('KMM-TRadaboost',predict1,y_test)
    # X_test, i), y_test)
    # 画ROC曲线和计算AUC,将返回的list归一化到0,1之间
    # min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    #
    #
    # predict = np.asarray(predict)
    # predict = predict.reshape(len(y_test), 1)
    # predict = min_max_scaler.fit_transform(predict)
    # predict=predict.tolist()
    # fpr1, tpr1, thresholds1 = roc_curve(y_test, predict)
    # roc_auc1 = auc(fpr1, tpr1)
    #
    # # predict1=np.asarray(predict1)
    # # predict1=predict1.reshape(len(y_test),1)
    # # predict1 = min_max_scaler.fit_transform(predict1)
    # # predict1=predict1.tolist()
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)
    # all_tpr = []
    # fpr, tpr, thresholds = roc_curve(y_test, predict1)  ##指定正例标签，pos_label = ###########在数之联的时候学到的，要制定正例
    # mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    # mean_tpr[0] = 0.0
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=1, label='vfkmm-tradaboost AUC = %0.2f'% roc_auc)
    # plt.plot(fpr1, tpr1, lw=1, label='tradaboost AUC = %0.2f'% roc_auc1)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'m--',c='#666666')
    # plt.show()



    plt.plot(reslisttempx, reslisttemp, marker='x', linestyle='dashed', linewidth=1, label="LR tradaboost")
    plt.plot(reslisttempx1, reslisttemp1, marker='o', linestyle='dashed', linewidth=1, label="vfkmm tradaboost")
    plt.plot(reslistx, reslist, marker='+', linestyle='dashed', linewidth=1,label="tradaboost")  # plt.plot(range(5,31,5), accuracy_scorelist[4:30:5],marker='x', linestyle='dashed',linewidth=1,label="vfkmm without eliminate")
    # plt.plot(range(5,31,5), accuracy_scorelist1[4:30:5], marker='o', linestyle='dashed',linewidth=1,label="vfkmm eliminate")
    plt.xlabel("迭代次数")
    plt.ylabel("score")
    plt.legend(loc="lower right")
    plt.show()
    #res = return_correct_rate(tradaboost(X_train, TFIDF[length:,:],y_train[:,None], merge_target_source_label[length:,None], X_test[0:200,:], 100),y_test[0:200,None])


from sklearn import svm
def naive_model_return_error(train, y, test,test_y):
    """implement a comparative method as a naive model"""
    #model = sklearn.linear_model.LogisticRegression(C=10000, penalty='l1', tol=0.0001)
    model = svm.SVC(C=131072,gamma=0.0001, kernel='rbf', probability=True)
    model.fit(train,y )
    preds = model.predict(test)
    c= 0
    for i in range(len(preds)):
        if preds[i] == test_y[i] :
            c+=1
    res = c/len(test_y)
    return res


def return_correct_rate(preds, target):
    c= 0
    for i in range(len(preds)):
        if preds[i] == target[i] :
            c+=1
    res = c/len(target)
    #print("准确率",np.mean(preds == target),'召回率',recall_score(preds,target),'F1分数',f1_score(preds,target))
    return res

def precision_score(y_true, y_pred):
    return ((y_true==1)*(y_pred==1)).sum()/(y_pred==1).sum()
def recall_score(y_true, y_pred):
    return ((y_true==1)*(y_pred==1)).sum()/(y_true==1).sum()
def f1_score(y_true, y_pred):
    num = 2*precision_score(y_true, y_pred)*recall_score(y_true, y_pred)
    deno = (precision_score(y_true, y_pred)+recall_score(y_true, y_pred))
    return num/deno
def split_test_classifier(clf, X, y,X_test,y_test):

    clf.fit(X, y)
    # predict
    y_predicted = clf.predict(X_test)
    # calculate percision
    print_classification_report('',y_predicted,y_test)
    print("准确率",np.mean(y_predicted == y_test),'召回率',recall_score(y_test,y_predicted),'F1分数',f1_score(y_test,y_predicted))
    return  np.mean(y_predicted == y_test)

def plot_results(i, results_list, labels_list):
    colors_list = ['red', 'blue', 'black', 'green', 'cyan', 'yellow']

    if not len(results_list) == len(labels_list):
        raise Exception

    for (result, label, color) in zip(results_list, labels_list, colors_list):
        plt.plot(i, result, color=color, lw=2.0, label=label)
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()


