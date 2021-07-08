# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:08:59 2021

@author: Sassan
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
import pylab

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

# Loading the files
file = loadmat('Gaussian.mat')
gaussian = file["gaussian"]
file2 = loadmat('GaussianPlus.mat')
gaussian_plus = file2["gaussianplus"]

#==========================================================
# Plot data and Ellipse (part 2)
#==========================================================
def ML_estimator_gaussian(data):
    """
    Take the gaussian data and compute its mean and covariance using the maximum
    likelihood estimator.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    (gaus_mean, gaus_cov)

    """
    X = data[:, 0] 
    Y = data[:, 1]
    gaus_mean = np.array([[np.mean(X)], [np.mean(Y)]])
    sum_gaus = 0
    for i in data:
        j = np.array([[i[0]],[i[1]]])
        sum_gaus += j.dot(j.transpose())
        gaus_cov = (1/(len(data))) * sum_gaus - gaus_mean.dot(gaus_mean.transpose())
    return (gaus_mean, gaus_cov)
#print(ML_estimator_gaussian(gaussian))

def Ellipse_generator_2D(data, n_std=1):
    """
    Take the data, return the ellipse represent its covariance matrix. The ellipse derived
    eigen-decomposition of the matrix and scaling the axes by obtained eigenvalues

    Parameters
    ----------
    data : np.ndarray
    n_std : Control the radius of the ellipse, which is the number of standard deviations 
        The default value is 1 which makes the ellipse enclose 68% of the points
        
    Returns 
    -------
    ellipse

    """
    gaus_mean, gaus_cov = ML_estimator_gaussian(data)
    eigVal , eigVec = np.linalg.eig(gaus_cov)
    angle=np.rad2deg(np.arccos(eigVec[0, 0]))
    width = np.sqrt(eigVal[0])*2*n_std
    height=np.sqrt(eigVal[1])*2*n_std
    
    return Ellipse([gaus_mean[0], gaus_mean[1]], width, height, angle, 
                   edgecolor='red', lw=4, facecolor='none')

# # Plot Gaussian data and ellipse
# fig, ax = plt.subplots()
# ell = Ellipse_generator_2D(gaussian, 1)
# ax.add_artist(ell)
# plt.scatter(gaussian[:, 0],gaussian[:, 1], c='gray', s=50)
# plt.scatter(ML_estimator_gaussian(gaussian)[0][0], ML_estimator_gaussian(gaussian)[0][1],
#             c="red", marker="o",s=150)
# plt.title('Gaussian')
# plt.style.use("fivethirtyeight")
# plt.xlabel('X')
# plt.ylabel('Y')


#==========================================================
# MAP for reduced dataset (part 3)
#==========================================================
def ReducedData_MAP_estimator(data, i, cov0, mean0):
    """
    Split the data up to given index i. Then, using the prior gaussian with given mean "mean0" 
    and covariance matrix "cov0", compute the MAP estimator of the reduced data.

    Parameters
    ----------
    data : np.ndarray

    i : scalar
        Determine where to cut the data
    cov0 : 
        covariance matrix of prior
    mean0 : 
        mean of the prior

    Returns
    -------
    MAP estimate of the mean

    """
    reduced_gaus = data[0:i]
    mean_n = np.mean(reduced_gaus, 0)
    covn = np.linalg.inv(np.linalg.inv(cov0)+i*np.linalg.inv(cov0))
    return covn@(i*(np.linalg.inv(cov0)@mean_n)+(np.linalg.inv(cov0)@mean0))

mean0 = np.array([2, 3]) # change this for mu0= [2,5] (and the title of the graph)
cov0 = ML_estimator_gaussian(gaussian)[1]
MAP2 = ReducedData_MAP_estimator(gaussian, 2, cov0, mean0)
MAP5 = ReducedData_MAP_estimator(gaussian, 5, cov0, mean0)
MAP10 = ReducedData_MAP_estimator(gaussian, 10, cov0, mean0)
MAPfull = ReducedData_MAP_estimator(gaussian, 100, cov0, mean0)
# plot
# plt.scatter(gaussian[:, 0], gaussian[:, 1], c='gray', s=50)
# ML2 = plt.scatter(ML_estimator_gaussian(gaussian[0:2])[0][0], ML_estimator_gaussian(gaussian)[0][1],
#                   c="red", marker="o",s=150, label="ML:2")
# ML5 = plt.scatter(ML_estimator_gaussian(gaussian[0:5])[0][0], ML_estimator_gaussian(gaussian)[0][1],
#                   c="blue", marker="o",s=150, label="ML:5")
# ML10 = plt.scatter(ML_estimator_gaussian(gaussian[0:10])[0][0], ML_estimator_gaussian(gaussian)[0][1],
#                   c="green", marker="o",s=150, label="ML:10")
# plt.scatter(MAP2[0], MAP2[1], c="red", marker="x", s=150, label="MAP:2")
# plt.scatter(MAP5[0], MAP5[1], c="blue", marker="x", s=150, label="MAP:5")
# plt.scatter(MAP10[0], MAP10[1], c="green", marker="x", s=150, label="MAP:10")
# plt.title('MAP and ML estimates with mu0=(2,3)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

#==========================================================
# ML of Gaussian plus (part 4)
#==========================================================


# Plot Gaussian plus data and ellipse
fig, ax = plt.subplots()
ell = Ellipse_generator_2D(gaussian_plus, 1)
ax.add_artist(ell)
plt.scatter(gaussian_plus[:, 0],gaussian_plus[:, 1], c='gray', s=50)
plt.scatter(ML_estimator_gaussian(gaussian_plus)[0][0], ML_estimator_gaussian(gaussian_plus)[0][1],
            c="red", marker="o",s=150)
plt.title('Gaussian Plus')
plt.style.use("fivethirtyeight")
plt.xlabel('X')
plt.ylabel('Y')










