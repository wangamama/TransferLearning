import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B / math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z, Z)
        kappa = np.sum(compute_rbf(Z, X), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K
x = 11*np.random.random(200)- 6.0
y = x**2 + 10*np.random.random(200) - 5
Z = np.c_[x, y]

x = 2*np.random.random(10) - 6.0
y = x**2 + 10*np.random.random(10) - 5
X = np.c_[x, y]



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# 样本重标记
lable_spy_a = np.zeros([200, 1])
lable_spy_s = np.ones([10, 1])


trans_data = np.concatenate((X, Z), axis=0)
trans_label = np.concatenate((lable_spy_s,lable_spy_a), axis=0)

X_train, X_test, y_train, y_test = train_test_split(trans_data, trans_label, test_size=0.33, random_state=42)
clf = LogisticRegression(penalty='l1',class_weight='balanced')
# gnb = BernoulliNB()
clf.fit(X_train, y_train)
print("LR的预测精度", metrics.confusion_matrix(y_test, clf.predict(X_test)))
print("LR的预测精度", metrics.accuracy_score(y_test, clf.predict(X_test)))





coef = clf.predict_proba(Z)[:, -1].tolist()


# coef = kernel_mean_matching(X, Z, kern='rbf', B=10)
# print(coef)
# print(coef.shape)
# print(Z.shape)
#
# plt.close()
# plt.figure()
# plt.scatter(Z[:,0], Z[:,1], color='black', marker='x')
# plt.scatter(X[:,0], X[:,1], color='red')
# plt.scatter(Z[:,0], Z[:,1], color='green', s=coef*10, alpha=0.5)
# plt.show()
# np.sum(coef > 1e-2)

#
#
# print(coef)
# print(Z.shape)
#
# plt.close()
# plt.figure()
#
# w=clf.coef_
# p=clf.intercept_
# print(w)
# print(p)
# x = np.mat(np.arange(min(Z[:,0]),max(Z[:,0]), 0.1))
# y = (-p[0]- w[0,0] * x) / w[0,1]
#
# coef = np.asarray(coef)
# plt.plot(x.transpose(), y.transpose())
# # plt.scatter(Z[:,0], Z[:,1], color='black', marker='x')
# plt.scatter(X[:,0], X[:,1], color='red', marker='x')
# plt.scatter(Z[:,0], Z[:,1], color='green',marker='o',s=coef*80, alpha=0.5)
# plt.show()
