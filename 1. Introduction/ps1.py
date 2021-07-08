# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:06:23 2021

@author: Sassan Mokhtar
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
#==========================================================
# Vectors and Matrices in Python 
#==========================================================

# Define a vector and a matrix
vec = np.array(np.random.random([3, 1]))
mat = np.array(np.random.random([3, 3]))

print("The randomly chosen 3*3 matrix is: \n", mat)
print("And the randomly chosen 3*1 vector is: \n", vec)

# Compute matrix-vector Multiplication
mult = mat.dot(vec)
print("The matrix-vector Multiplication is: \n", mult)

# Invert the matrix
inv = np.linalg.inv(mat)
print("The inverse of the matrix is: \n", inv)

#==========================================================
# Loading and ploting 2 datasets 
#==========================================================
# Loading the files
file = loadmat('Gaussian.mat')
gaussian = file["gaussian"]
file2 = loadmat('GaussianPlus.mat')
gaussian_plus = file2["gaussianplus"]

# Plotting two datasets

Xval1 = gaussian[:, 0] 
Yval1 = gaussian[:, 1]
Xval2 = gaussian_plus[:, 0]
Yval2 = gaussian_plus[:, 1]

plt.subplot(1, 2, 1)
plt.plot(Xval1, Yval1, "o", color='black')
plt.title('Gaussian')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.plot(Xval2, Yval2, "o", color='red')
plt.title('Gaussian Plus')
plt.xlabel('X')
plt.style.use("fivethirtyeight")
plt.tight_layout(pad=.02)
plt.show()



