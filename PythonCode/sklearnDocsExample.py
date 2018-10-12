"""
Learning to use the scikit-learn K-means clustering algorithm
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

print(__doc__)


iris = datasets.load_iris()
digits = datasets.load_digits()

# The .data member is an (n-samples, n_features) array
print(digits.data)
# digits.target gives the ground truth for the digit dataset
print(digits.target)
# Below is an example of accessing the original data
print(digits.data[0])


'''
A Note on some important modules:
-> The sklearn.feature_extraction module can be used to extract desirable features from raw data, so that it can be
   used by sklearn's clustering algorithms.
-> sklearn.metrics.pairwise module can be used to obtain matrices of the shape [n_samples, n_samples] to be used with
   AffinityPropogation, SpectralClustering and DBSCAN algorithms.
'''
'''
A Note on K-means:
-> For the K-means algorithm, using the parameter init='k-means++' will use the k-means++ initialization scheme which
   leads to provably better results than random initialization (reduces the likelihood of converging to a local
   minimum.
-> Using the sample_weight parameter will assign more weight to some samples.
-> A parameter can be given to allow K-means to be run in parallel, called n_jobs. Giving this parameter a positive 
   value uses that many processors (default: 1). A value of -1 uses all available processors, with -2 using one
   less, an do on. Parallelization generally speeds up computation at the cost of memory (in this case, multiple
   copies of centroids need to be stored, one for each job)
'''

# Example, taken from: https://stackabuse.com/k-means-clustering-with-scikit-learn/

# Data points:
X = np.array([[5,3],
              [10,15],
              [15,12],
              [24,10],
              [30,45],
              [85,70],
              [71,80],
              [60,78],
              [55,52],
              [80,91]])

# Visualize the points:
plt.scatter(X[:,0],X[:,1], label='True Position')

# Create Clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

# Visualize the clusters:
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_,cmap='rainbow')
plt.show()

# Fit 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='black')
plt.show()

# Create 2 new points
Y = np.array([[49,63],
              [23,77]])

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_,cmap='rainbow')
plt.scatter(Y[:,0],Y[:,1], color='black')
plt.show()

Yweights = kmeans.predict(Y)
print(Yweights)

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_,cmap='rainbow')
plt.scatter(Y[:,0],Y[:,1], c=Yweights,cmap='rainbow')
plt.show()
