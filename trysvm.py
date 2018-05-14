from BinarySVM import BinarySVM
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])
classifier = BinarySVM(x_vals, y_vals)
classifier.fit()
