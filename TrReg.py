
import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from TR.TrAdaboostreg import *
from TR.TTemp import *
from TR.Mutisource import *
##=============================================================================

#                                Example 1
##=============================================================================

# 1. define the data generating function
def response(x, d, random_state,rng_temp):
    """
    x is the input variable
    d controls the simularity of different tasks
    """
    # a1 = np.random.normal(1, 0.1 * d)
    # a2 = np.random.normal(1, 0.1 * d)
    # b1 = np.random.normal(1, 0.1 * d)
    # b2 = np.random.normal(1, 0.1 * d)
    # c1 = np.random.normal(1, 0.05 * d)
    # c2 = np.random.normal(1, 0.05 * d)
    a1 = rng_temp.normal(1, 0.1 * d)
    a2 = rng_temp.normal(1, 0.1 * d)
    b1 = rng_temp.normal(1, 0.1 * d)
    b2 = rng_temp.normal(1, 0.1 * d)
    c1 = rng_temp.normal(1, 0.05 * d)
    c2 = rng_temp.normal(1, 0.05 * d)
    y = a1 * np.sin(b1 * x + c1).ravel() + a2 * np.sin(b2 * 6 * x + c2).ravel() + random_state.normal(0, 0.1,
                                                                                                      x.shape[0])
    return y


# ==============================================================================

#     2. decide the degree of similarity of multiple data sources using d

d = 2
# ==============================================================================
rng = np.random.RandomState(11)
rng_temp = np.random.RandomState(11)
# 3.1 create source data and target data
n_source1 = 50
#x_source1 = np.linspace(0, 6, n_source1)[:, np.newaxis]

x_source1=6*rng_temp.random_sample(n_source1)[:, np.newaxis]
y_source1 = response(x_source1, 0.5, rng,rng_temp)
n_source2 = 100
#x_source2 = np.linspace(0, 6, n_source2)[:, np.newaxis]
x_source2=6*rng_temp.random_sample(n_source1)[:, np.newaxis]
y_source2 = response(x_source2, 2.0, rng,rng_temp)
n_source3 = 100
#x_source3 = np.linspace(0, 6, n_source3)[:, np.newaxis]
x_source3=6*rng_temp.random_sample(n_source1)[:, np.newaxis]
y_source3 = response(x_source3, 6.5, rng,rng_temp)
n_source4 = 100
#x_source4 = np.linspace(0, 6, n_source4)[:, np.newaxis]
x_source4=6*rng_temp.random_sample(n_source1)[:, np.newaxis]
y_source4 = response(x_source4, 6.0, rng,rng_temp)
n_source5 = 100
#x_source5 = np.linspace(0, 6, n_source5)[:, np.newaxis]
x_source5=6*rng_temp.random_sample(n_source1)[:, np.newaxis]
y_source5 = response(x_source5, 5.0, rng,rng_temp)

# 3.2 create target data (n_target_train and n_target_test are the sample size of train and test datasets)
d=0.05
rng_temp2 = np.random.RandomState(43)
# a1 = np.random.normal(1, 0.1 * d)
# a2 = np.random.normal(1, 0.1 * d)
# b1 = np.random.normal(1, 0.1 * d)
# b2 = np.random.normal(1, 0.1 * d)
# c1 = np.random.normal(1, 0.05 * d)
# c2 = np.random.normal(1, 0.05 * d)
a1 = rng_temp2.normal(1, 0.1 * d)
a2 = rng_temp2.normal(1, 0.1 * d)
b1 = rng_temp2.normal(1, 0.1 * d)
b2 = rng_temp2.normal(1, 0.1 * d)
c1 = rng_temp2.normal(1, 0.05 * d)
c2 = rng_temp2.normal(1, 0.05 * d)

# target_train
# ==============================================================================

n_target_train = 40

# ==============================================================================
#x_target_train = np.linspace(0, 6, n_target_train)[:, np.newaxis]

x_target_train=6*rng_temp2.random_sample(n_target_train)[:, np.newaxis]


y_target_train = a1 * np.sin(b1 * x_target_train + c1).ravel() + a2 * np.sin(
    b2 * 6 * x_target_train + c2).ravel() + rng.normal(0, 0.1, x_target_train.shape[0])

# target_test
n_target_test = 600
#x_target_test = np.linspace(0, 6, n_target_test)[:, np.newaxis]

x_target_test=6*rng_temp2.random_sample(n_target_test)[:, np.newaxis]
y_target_test = a1 * np.sin(b1 * x_target_test + c1).ravel() + a2 * np.sin(
    b2 * 6 * x_target_test + c2).ravel() + rng.normal(0, 0.1, x_target_test.shape[0])


X = np.concatenate((x_source1, x_source2, x_source3, x_source4, x_source5))
y = np.concatenate((y_source1, y_source2, y_source3, y_source4, y_source5))


# ==============================================================================
from sklearn import neighbors

clf = DecisionTreeRegressor(max_depth=3)
#clf = neighbors.KNeighborsRegressor()

clf.fit(x_target_train,y_target_train)
predict1=clf.predict(x_target_test)
mse_twostageboost = mean_squared_error(y_target_test, predict1)
print("MSE of tree:", mse_twostageboost)
print("r2 of tree:", r2_score(y_target_test, predict1))
# ==============================================================================



xlist=[x_source1,x_source2,x_source3,x_source4,x_source5]
ylist=[y_source1, y_source2, y_source3, y_source4, y_source5]
reslisttempx2 = []
reslisttemp2 = [];
for i in range(4, 50, 1):
    predict = Mutisource_tradaboost(
    x_target_train, xlist, y_target_train, ylist,  x_target_test, i,5,4*100+50)
    mse_twostageboost = mean_squared_error(y_target_test, predict)
    reslisttemp2.append(r2_score(y_target_test, predict))
    reslisttempx2.append(i)
print("r2 of tradaboost:", reslisttemp2)



print("MSE of muti:", mse_twostageboost)
print("r2 of muti:", r2_score(y_target_test, predict))
reslisttempx1 = []
reslisttemp1 = [];
for i in range(4, 50, 1):
    predict = tradaboost(
        x_target_train, X, y_target_train, y, x_target_test, y_target_test, i, True)
    mse_twostageboost = mean_squared_error(y_target_test, predict)
    reslisttemp1.append(r2_score(y_target_test, predict))
    reslisttempx1.append(i)
print("r2 of tradaboost:", reslisttemp1)
#
#
# reslisttempx2 = []
# reslisttemp2 = [];
# for i in range(20, 300, 10):
#     predict = tradaboost(
#         x_target_train, X, y_target_train, y, x_target_test, y_target_test, i, False)
#     mse_twostageboost = mean_squared_error(y_target_test, predict)
#     reslisttemp2.append(r2_score(y_target_test, predict))
#     reslisttempx2.append(i)
# print("r2 of tradaboost:", reslisttemp2)
#
plt.plot(reslisttempx1, reslisttemp1, marker='*', linestyle='dashed', linewidth=1, label="tradaboost")
plt.plot(reslisttempx2, reslisttemp2, marker='+', linestyle='dashed', linewidth=1, label="mutisource vfkmm-tradaboost")
plt.plot(range(4, 50, 1), [0.70]*46, marker='_', linestyle='dashed', linewidth=1, label="baseline")

plt.xlabel("Iterations")
plt.ylabel("score")
plt.legend(loc="lower right")
plt.show()



predict = tradaboost(
    x_target_train, X, y_target_train, y,  x_target_test,y_target_test, 300,True)
mse_twostageboost = mean_squared_error(y_target_test, predict)
print("MSE of tradaboost:", mse_twostageboost)
print("r2 of tradaboost:", r2_score(y_target_test, predict))


#
#
predict2 = tradaboost(
    x_target_train, X, y_target_train, y,  x_target_test,y_target_test, 300,False)
mse_twostageboost = mean_squared_error(y_target_test, predict2)
print("MSE of tradaboost margin:", mse_twostageboost)
print("r2 of tradaboost: margin", r2_score(y_target_test, predict2))




# 4.4 Plot the results
plt.figure()
plt.scatter(x_target_train, y_target_train, c="k", label="target_train")
plt.plot(x_target_test, y_target_test, c="b", label="target_test", linewidth=0.5)
plt.plot(x_target_test, predict1, c="r", label="AdaBoostRegressor", linewidth=2)
# plt.plot(x_target_test, predict, c="g", label="VFKMM-TrAdaBoost without margin", linewidth=2)
# plt.plot(x_target_test, predict2, c="y", label="VFKMM-TrAdaBoost", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="lower left")
plt.title("mutisource VFKMM-TrAdaBoost Regressor")
plt.legend()
# plt.show()
