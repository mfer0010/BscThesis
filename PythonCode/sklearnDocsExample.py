from sklearn import datasets

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

