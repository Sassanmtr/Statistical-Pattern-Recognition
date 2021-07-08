# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:56:26 2021

@author: Sassan
"""
#===================================================
# Considerations:
# performance of K mean is better (I used silhouette score)  !!
# How to change the initial condition
#===================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from numpy import random
from scipy.stats import multivariate_normal
#  load the file
file = loadmat('GaussianPlus.mat')
gaussian_plus = file["gaussianplus"]
Xval = gaussian_plus[:, 0]
Yval = gaussian_plus[:, 1]
#####################################
# Built-in K-mean
#####################################
kmeans = KMeans(init="random", n_clusters=2, n_init=10, max_iter=300, random_state=0)
kmeans.fit(gaussian_plus)
centroids = kmeans.cluster_centers_
score_Kmean = silhouette_score(gaussian_plus, kmeans.labels_)

#The k-means plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fte_colors = {0: "#008fd5", 1: "#fc4f30",}
km_colors = [fte_colors[label] for label in kmeans.labels_]
ax.scatter(Xval, Yval, c=km_colors)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = "k")
plt.title('Built-in K means')
#####################################
# Built-in EM
#####################################
em = GaussianMixture(n_components=2, covariance_type='full', 
                     n_init=10, max_iter=300, random_state=0).fit(gaussian_plus)
label_EM = em.predict(gaussian_plus)
score_EM = silhouette_score(gaussian_plus, label_EM)

# # The EM plot
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# fte_colors = {0: "#008fd5", 1: "#fc4f30",}
# em_colors = [fte_colors[label] for label in label_EM]
# ax.scatter(Xval, Yval, c=em_colors)
# plt.scatter(em.means_[:, 0], em.means_[:, 1], s = 80, color = "k")
# plt.title('Built-in EM')

#####################################
# Implementation of K-mean (Naive)
#####################################
random.seed(0)
data = gaussian_plus
n_clusters = 2
max_iter = 1000
def Kmeans_implemented(data, max_iter, n_clusters):
    """
    Perform Simplified Version of K means algorithm
    LIMITATIONS: 
        1) Only works for n_cluster = 2
        2) No breaking strategy --> It finishes when all the iterations are done
    Parameters
    ----------
    data : ndarray
        DESCRIPTION.
    max_iter : int
        maximum number of iteration
    n_clusters : int (the function only works for n_clusters=2)
        Number of desired clusters

    Returns
    -------
    centroid : ndarray
        means of clusters
    cluster1 : ndarray
        contains of points assigned to cluster 1
    cluster2 : ndarray
        contains of points assigned to cluster 2

    """
    counter = 0
    x_bound = [data[:,0].min(), data[:,0].max()] 
    y_bound = [data[:,1].min(), data[:,1].max()]
    lower = np.array([x_bound[0], y_bound[0]])
    upper = np.array([x_bound[1], y_bound[1]])
    centroid = {}
    for i in range(n_clusters):
        centroid[i+1] = random.uniform(lower, upper)
    counter = 0
    while counter < max_iter:     
        # E-step
        cluster1 = []
        cluster2 = []
        for point in data:
            if np.linalg.norm(centroid[1] - point) < np.linalg.norm(centroid[2] - point):
                cluster1.append(point)
            else:
                cluster2.append(point)
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
        # M-step
        centroid[1] = np.mean(cluster1,0)
        centroid[2] = np.mean(cluster2,0)
        # loop updates
        counter +=1
    return (centroid, cluster1, cluster2)

# centroid_Kmean, cluster1, cluster2 = Kmeans_implemented(data, max_iter, n_clusters)
# # The plot of Implemented Kmeans
# centroid_array = []
# for i in centroid_Kmean:
#     centroid_array.append(centroid_Kmean[i])
# centroid_array = np.array(centroid_array)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.title('Implemented K-means')
# plt.scatter(data[:,0], data[:,1])
# plt.scatter(centroid_array[:, 0] , centroid_array[:, 1] , s = 200, marker="x",  color = "r")
# ax.plot(cluster1[:, 0], cluster1[:, 1], "o", color='orange', alpha=0.8)
# ax.plot(cluster2[:, 0], cluster2[:, 1], "o", color='green', alpha=0.2)
# plt.show()


#####################################
# Implementation of EM
#####################################
random.seed(0)
data = gaussian_plus
n_clusters = 2
max_iter = 300
def EM_implemented(data, max_iter, n_clusters, mix=0.5):
    """
    Perform Simplified Version of EM algorithm
    LIMITATIONS: 
        1) Only works for n_cluster = 2
        2) No breaking strategy --> It finishes when all the iterations are done
    Parameters
    ----------
    data : ndarray
        DESCRIPTION.
    max_iter : int
        maximum number of iteration
    n_clusters : int (the function only works for n_clusters=2)
        Number of desired clusters

    Returns
    -------
    mean1, mean2 : ndarray
        means of clusters
    final_clus1, final_clus2 : ndarray
        contains of points assigned to cluster 1 and 2 respectively.

    """
    # Initialization
    mix = 0.5
    x_bound = [data[:,0].min(), data[:,0].max()] 
    y_bound = [data[:,1].min(), data[:,1].max()]
    lower = np.array([x_bound[0], y_bound[0]])
    upper = np.array([x_bound[1], y_bound[1]])
    mu1 = random.uniform(lower, upper)
    mu2 = random.uniform(lower, upper)
    cov1, cov2 = np.array([[2, 1],[1, 2]])  
    count = 0
    while count < max_iter:
        # E-step (compute the responsibility of each cluster for each point)
        expectations = []
        for point in data:
            gamma1 = mix * multivariate_normal.pdf(point, mean=mu1, cov=cov1)
            gamma2 = (1 - mix) * multivariate_normal.pdf(point, mean=mu2, cov=cov2)
            expectations.append([point, gamma1/(gamma1+gamma2), gamma2/(gamma1+gamma2)])
        # M-step (Update variables)
        N1 = 0
        N2 = 0
        temp1 = 0 
        temp2 = 0
        for i in expectations:
            N1 += i[1]
            N2 += i[2]
            temp1 += i[0] * i[1]
            temp2 += i[0] * i[2]
        mu1 = temp1/N1
        mu2 = temp2/N2
        cov1=0
        cov2=0
        for j in expectations:
            temp_sum1 = j[0] - mu1
            temp_sum2 = j[0] - mu2
            cov1 += j[1] * np.array([temp_sum1[0]* temp_sum1, temp_sum1[1]* temp_sum1])
            cov2 += j[2] * np.array([temp_sum2[0]* temp_sum2, temp_sum2[1]* temp_sum2])
        cov1 = cov1/N1
        cov2 = cov2/N2
        mix = N1/len(data)
        # loop updates
        count += 1
    final_clus1 = []
    final_clus2 = []
    for point in expectations:
        if point[1] < point[2]:
            final_clus2.append(point[0])
        else:
            final_clus1.append(point[0])
    final_clus1 = np.array(final_clus1)
    final_clus2 = np.array(final_clus2)
    mean1 = np.array([mu1[0], mu2[0]])
    mean2 = np.array([mu1[1], mu2[1]])
    return (mean1, mean2, final_clus1, final_clus2)

mean1, mean2, final_clus1, final_clus2 = EM_implemented(data, max_iter, n_clusters, mix=0.5)

# # plot of implemented EM
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.title('Implemented EM')
# plt.scatter(data[:,0], data[:,1])
# plt.scatter(mean1 , mean2 , s = 200, marker="x",  color = "r")
# ax.plot(final_clus1[:, 0], final_clus1[:, 1], "o", color='orange', alpha=0.8)
# ax.plot(final_clus2[:, 0], final_clus2[:, 1], "o", color='green', alpha=0.2)
# plt.show()























