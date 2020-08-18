import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#mat = loadmat('ex7data2.mat');
#X = mat['X'];
#centroids = np.array([[6,2],[3,3],[8,5]])
#temp = np.zeros(k)

def findClosestCentroids(X, centroids): # for each example find its closest centroid
    m = X.shape[0]    # m is number of training example m=300 
    k = centroids.shape[0]  # k is number of centroid k =3
    idx = np.zeros(m)  # row vector of dimension (300,)
    idx[0] = 1    
    temp = np.zeros(k) #temp is row vector of dimension is (3,)
    for i in range(1):
        for j in range(3):
            dist = np.sum((X[i,:] - centroids[j,:])**2) # square root nhi liya because compare hi karna hai 
            temp[j] = dist  # each example ke liye teeno centroid se distance dekhna hoga na but each inner iteration me to ek hi example se distance calculate ho raha hai so temp me store kar raha hai 
        idx[i] = np.argmin(temp) # temp[0] , temp[1] , temp[2] me minimum value temp[1] ka hai so 1 store hoga at 0th index of idx
    return idx

















mat = loadmat('ex7data2.mat');
X = mat['X'];
plt.plot(X[:,0],X[:,1],'k+')
# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = findClosestCentroids(X, initial_centroids)
print("Closest centroids for the first 3 examples:\n",idx[0:3])