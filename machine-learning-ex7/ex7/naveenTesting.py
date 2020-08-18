import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat



def find_closest_centroids(X, centroids): # for each example find its closest centroid
    m = X.shape[0]    # m is number of training example m=300 
    k = centroids.shape[0]  # k is number of centroid k =3
    idx = np.zeros(m)  # row vector of dimension (300,)
    idx[0] = 1    
    temp = np.zeros(k) #temp is row vector of dimension is (3,)
    for i in range(m):
        for j in range(k): # let m = 1 only one training example     #centroids = np.array([[6,2],[3,3],[8,5]])
            dist = np.sum((X[i,:] - centroids[j,:])**2) # square root nhi liya because compare hi karna hai 
            temp[j] = dist  # each example ke liye teeno centroid se distance dekhna hoga na but each inner iteration me to ek hi example se distance calculate ho raha hai so temp me store kar raha hai 
        idx[i] = np.argmin(temp) # temp[0] , temp[1] , temp[2] me minimum value temp[1] ka hai so 1 store hoga at 0th index of idx
    return idx                   #argmin returns the index of minimum value


def compute_centroids(X, idx, k):  
    m, n = X.shape
    centroids = np.zeros((k, n))# dimension of 3 x 2  # jitna feature hoga utna hi cordinate hoga , two feature x1 and x2 means only x , y cordinate , x1,x2,x3 then x,y,z cordinate 
    for i in range(k):
        indices = np.where(idx == i) # idx[0] stores information that  0th example belongs to which centroid idx[3] = X[3] belongs to which centroid 
        # first loop me i==0 hai and where will fetch all the index where value == 0 menas it will fetch all those feature who belongs to centroid 0
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel() # axis =1 means sum coloum wise karo tabhi na all x gets suumed and all y get summed 
    return centroids

def run_k_means(X, initial_centroids, max_iters):  
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):   # after a given number of iteration , k means will converge 
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids












mat = loadmat('ex7data2.mat');
X = mat['X'];
# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = find_closest_centroids(X, initial_centroids)
print("Closest centroids for the first 3 examples:\n",idx[0:3])

centroids = compute_centroids(X, idx, 3)
print("Centroids computed after initial finding of closest centroids:\n", centroids)


idx, centroids = run_k_means(X, initial_centroids, 10)


# plotting the result 

cluster1 = X[np.where(idx == 0)[0],:]  
cluster2 = X[np.where(idx == 1)[0],:]  
cluster3 = X[np.where(idx == 2)[0],:]
fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(cluster1[:,0], cluster1[:,1], color='r', label='Cluster 1')  
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')  
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')  
ax.legend()