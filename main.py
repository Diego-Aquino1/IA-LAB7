import pylab as pl
import numpy as np
from sklearn import datasets

X, Y = datasets.make_regression(n_samples=5, n_features=1, noise=20)

for j in range(X.size):
    print(str(X[j][0]) + "\t" + str(Y[j]))
pl.scatter(X, Y)
pl.show()