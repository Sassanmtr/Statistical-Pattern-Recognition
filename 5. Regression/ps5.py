# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:36:16 2021

@author: Sassan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import linalg
from sklearn.mixture import GaussianMixture

# importing the data
data = np.load('regression.npz')
X = data["arr_0"]

#=============================================

# visualizing the data
def Data_visualizer(X):
    plt.scatter(X[:, 0], X[:, 1], c = "black")
    plt.xlabel("X")
    plt.ylabel("y")
#Data_visualizer(X)    

#==============================================

# Polynomial Regression Using Python Built-in Functions
X_train, X_test, y_train, y_test = train_test_split(X[:, 0], X[:, 1], test_size=0.2, random_state=0)
def Regression_builtin(X_train, X_test, y_train, y_test):
    k = 5
    model = []
    MSE_train = []
    MSE_test = []
    for i in range(k):
        model.append(np.poly1d(np.polyfit(X_train, y_train, i+1)))
        y_hat = model[i](X_train)
        MSE_train.append(mean_squared_error(y_hat, y_train))
        y_bar = model[i](X_test)
        MSE_test.append(mean_squared_error(y_bar, y_test))
    x = np.linspace(0.1, 1, 101)
    plt.title("Regression with built in function")
    plt.style.use("fivethirtyeight")
    plt.scatter(X[:, 0], X[:, 1], s=100, c = "black")
    plt.scatter(X_test, y_test, s=100, c="purple")
    for i in range(len(model)):
        plt.plot(x, model[i](x), label="deg {}".format(i+1))
    plt.legend()
    plt.style.use("fivethirtyeight")
    plt.xlabel("X")
    plt.ylabel("y")
    return (MSE_train, MSE_test)
#MSE_reg_train, MSE_reg_test = Regression_builtin(X_train, X_test, y_train, y_test)

#==============================================
# Part 1 (Regression using ML estimator)
#==============================================
def phi_vec(X, m):
    T = X[:,1]
    phi = []
    for i in range(m+1):
        phi.append(X[:,0] ** i)
    phi = np.array(phi).T  
    factor1 = linalg.pinv2(np.matmul(phi.T, phi))
    factor2 = np.matmul(factor1, phi.T)
    W = factor2.dot(T)
    return  (phi, W[::-1])


def Poly_reg_imp(W, m):
    model = np.poly1d(W)
    x = np.linspace(0.1, 1, 101)
    plt.scatter(X[:, 0], X[:, 1], s=100, c = "black")
    plt.plot(x, model(x), label="deg {}".format(m))
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    
# m1 = 4
# W1 = phi_vec(X, m1)[1]
# Poly_reg_imp(W1, m1)


#==============================================
# Part 2 (Regression using MAP estimator)
#==============================================


def reg_MAP_imp(X, alpha, beta, m2):
    W2 = phi_vec(X, m2)[1].reshape(1, -1)
    phi = phi_vec(X, m2)[0]
    model = np.poly1d(W2.flatten())
    S = alpha * np.identity(m2 + 1) + beta * np.dot(phi.T,phi)
    factor1 = beta * linalg.inv(S)
    factor2 = np.dot(phi.T, X[:,1].reshape(-1, 1))
    m = np.dot(factor1, factor2)
    mean = np.dot(phi, m)
    factor3 = []
    for i in range(phi.shape[0]):
        a = phi[i].reshape(1, -1)
        b = np.dot(a, linalg.inv(S))
        factor3.append(np.dot(b, a.T))
    factor3 = np.array(factor3).reshape(-1, 1)
    var = (1 / beta) + factor3
    upper = mean + var
    lower = mean - var
    return (model, mean, lower, upper)

# plot the resulting MAP estimator
m2 = 5
beta = 10
alpha = 0.1
model, mean, lower, upper = reg_MAP_imp(X, alpha, beta, m2)
x = np.linspace(0.1, 1, 101)
plt.scatter(X[:, 0], X[:, 1], s=100, c = "black", label="data points")
plt.scatter(X[:, 0], mean, marker="*", s=50, c = "red", label="prediction mean")
plt.scatter(X[:, 0], lower, marker="+", s=50, c = "purple", label="prediction mean-+variance")
plt.scatter(X[:, 0], upper, marker="+", s=50, c = "purple")
plt.plot(x, model(x), label="deg {} ML reg".format(m2))
plt.legend()
plt.xlabel("X")
plt.ylabel("y")

#==============================================
# Part 3 (Optimal HPs using Evidence Approximation)
#==============================================
m3 = 4
max_iter = 20
beta = 1
alpha = 0.0001
counter = 0
T = X[:,1]
W = phi_vec(X, m3)[1].reshape(1, -1)
phi = phi_vec(X, m3)[0]
while counter < max_iter:
    # E-step
    S_N = alpha * np.identity(m3 + 1) + beta * np.dot(phi.T,phi)
    factor1 = beta * linalg.inv(S_N)
    factor2 = np.dot(phi.T, X[:,1].reshape(-1, 1))
    m_N = np.dot(factor1, factor2)
    print("m_N: ", np.dot(m_N.T, m_N))
    # M-step
    eigs = linalg.eig(beta * np.dot(phi.T,phi))[0].reshape(-1, 1)
    gamma = sum(np.divide(eigs, eigs + alpha))
    alpha = np.absolute(gamma / np.dot(m_N.T, m_N))
    beta_inv = sum((T - np.dot(W, phi.T) ** 2).flatten()) / (phi.shape[0] - gamma)
    beta = np.absolute(1 / beta_inv)
    # Loop updates
    counter += 1

print(alpha, beta)




















