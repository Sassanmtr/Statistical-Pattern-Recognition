# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:38:39 2021

@author: Sassan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pylab
import sklearn.model_selection as model_selection
#set line width
pylab.rcParams['lines.linewidth'] = 4
#set font size for titles 
pylab.rcParams['axes.titlesize'] = 20
#set font size for labels on axes
pylab.rcParams['axes.labelsize'] = 20
#set size of numbers on x-axis
pylab.rcParams['xtick.labelsize'] = 16
#set size of numbers on y-axis
pylab.rcParams['ytick.labelsize'] = 16
#set size of ticks on x-axis
pylab.rcParams['xtick.major.size'] = 7
#set size of ticks on y-axis
pylab.rcParams['ytick.major.size'] = 7
#set size of markers, e.g., circles representing points
#set numpoints for legend
pylab.rcParams['legend.numpoints'] = 1
#====================
# import the data
#====================
data = np.load('dataset.npz')
lst = data.files
for item in lst:
    x1 = data[item][:, 0]
    x2 = data[item][:, 1]
    y = data[item][:, 2]
#====================
# plot the data
#====================
def Plot_data(x1, x2, y):    
    X1 = []
    X2 = []
    X3 = []
    for i in range(len(y)):
        if y[i] == 1.0:
            X1.append(np.array([x1[i], x2[i]]))
        elif y[i] ==2.0:
            X2.append(np.array([x1[i], x2[i]]))
        else:
            X3.append(np.array([x1[i], x2[i]]))
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)    
    fig = plt.figure()
    plt.style.use("fivethirtyeight")
    plt.scatter(X1[:, 0] , X1[:, 1] , s = 30, marker="*",  color = "g", label="label 1")
    plt.scatter(X2[:, 0] , X2[:, 1] , s = 20, marker="o",  color = "b", label="label 2")
    plt.scatter(X3[:, 0] , X3[:, 1] , s = 20, marker="s",  color = "r", label="label 3")
    plt.legend()
    
#Plot_data(x1, x2, y)    

#====================
# split the data evenly and predict the labels by KNN
#====================
def KNN(x1, x2, y, k, train_size):
    X = []
    for i in range(len(x1)):
        X.append(np.array([x1[i],x2[i]]))
    X = np.array(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split (
        X, y, train_size=train_size,test_size=1-train_size, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_bar = neigh.predict(X_test)
    return y_bar, X_train, X_test, y_train, y_test


y_bar, X_train, X_test, y_train, y_test = KNN(x1, x2, y, k=1, train_size=0.5)

#====================
# Classification error
#====================
def KNN_error(X_test, y_test, y_bar):
    err = 0
    mclass = []
    for i in range(len(y_test)):
        if y_test[i] != y_bar[i]:
            mclass.append(X_test[i])
            err += 1
    mclass = np.array([mclass])
    return mclass, (err/len(y_test)) * 100

mclass, err = KNN_error(X_test, y_test, y_bar)

# plot misclassified data points
fig = plt.figure()
plt.style.use("fivethirtyeight")
plt.scatter(X_train[:, 0] , X_train[:, 1] , s = 20,  color = "b", label="Training")
plt.scatter(X_test[:, 0] , X_test[:, 1] , s = 20,  color = "g", label="Test")
plt.scatter(mclass[0][:, 0] , mclass[0][:, 1] , s = 50, marker="x", color = "r", label="Misclass")
plt.legend()

print("The average classification error rate: {:.2f} %".format(err))

#====================
# Bias-variance tradeoff
#====================


# train_ratio = np.linspace(.2,.8,41)
# number_neigh = np.linspace(1,20, 20)
# dict = {}
# mean = []
# variance = []
# for k in number_neigh:
#     k = int(k)
#     dict[k] = []
#     for i in train_ratio:
#         y_bar, X_train, X_test, y_train, y_test = KNN(x1, x2, y, k=k, train_size=i)
#         dict[k].append(KNN_error(X_test, y_test, y_bar)[1])
#     mean.append(np.mean(dict[k]))
#     variance.append(np.var(dict[k]))    
# mean = np.array(mean)
# variance = np.array(variance)
# error = mean + variance


# plt.plot(number_neigh, mean, color = "green", label="Bias")
# plt.plot(number_neigh, variance, color = "blue",label="Variance")
# plt.plot(number_neigh, error, color = "red",label="Bias+variance")
# plt.legend()

















