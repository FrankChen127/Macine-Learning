import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics


def bench_k_means(estimator, name, data): # cluster performance evaluation
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


digits = load_digits()
data = scale(digits.data) # scale() Standardize a dataset along any axis.
y = digits.target

model = KMeans(n_clusters=10, init="random", n_init=10)
bench_k_means(model, "1", data)

