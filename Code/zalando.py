## Example - Labeling Zalando pics ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
d1 = np.genfromtxt(path + 'zalando1.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d2 = np.genfromtxt(path + 'zalando2.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d3 = np.genfromtxt(path + 'zalando3.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d4 = np.genfromtxt(path + 'zalando4.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d5 = np.genfromtxt(path + 'zalando5.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d6 = np.genfromtxt(path + 'zalando6.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d7 = np.genfromtxt(path + 'zalando7.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
data = np.concatenate([d1, d2, d3, d4, d5, d6, d7])
np.unique(data['label'], return_counts=True)

# Target vector and feature matrix #
y = data['label']
from numpy.lib.recfunctions import structured_to_unstructured
X = structured_to_unstructured(data)[:, 1:]
X.shape
np.unique(X)

# Plotting the images #
from matplotlib import pyplot as plt
plt.gray()
X0 = X[y == 0, :]
plt.imshow(255 - X0[0, :].reshape(28,28));
X1 = X[y == 1, :]
plt.imshow(255 - X1[0, :].reshape(28,28));
X2 = X[y == 2, :]
plt.imshow(255 - X2[0, :].reshape(28,28));
X3 = X[y == 3, :]
plt.imshow(255 - X3[0, :].reshape(28,28));
X4 = X[y == 4, :]
plt.imshow(255 - X4[0, :].reshape(28,28));
X5 = X[y == 5, :]
plt.imshow(255 - X5[0, :].reshape(28,28));
X6 = X[y == 6, :]
plt.imshow(255 - X6[0, :].reshape(28,28));
X7 = X[y == 7, :]
plt.imshow(255 - X7[0, :].reshape(28,28));
X8 = X[y == 8, :]
plt.imshow(255 - X8[0, :].reshape(28,28));
X9 = X[y == 9, :]
plt.imshow(255 - X9[0, :].reshape(28,28));

# Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)

# Random forest classifier #
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(max_leaf_nodes=256, n_estimators=100, random_state=0)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)

# Multilayer perceptron classifier #
from sklearn.neural_network import MLPClassifier
mlpclf = MLPClassifier(hidden_layer_sizes=(32))
mlpclf.fit(X_train, y_train)
round(mlpclf.score(X_train, y_train), 3), round(mlpclf.score(X_test, y_test), 3)

# Normalization #
X = X/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)
rfclf = RandomForestClassifier(max_leaf_nodes=256, n_estimators=100, random_state=0)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)
mlpclf = MLPClassifier(hidden_layer_sizes=(32), random_state=0)
mlpclf.fit(X_train, y_train)
round(mlpclf.score(X_train, y_train), 3), round(mlpclf.score(X_test, y_test), 3)
mlpclf = MLPClassifier(hidden_layer_sizes=(32), max_iter=100, random_state=0)
mlpclf.fit(X_train, y_train)
round(mlpclf.score(X_train, y_train), 3), round(mlpclf.score(X_test, y_test), 3)
mlpclf = MLPClassifier(hidden_layer_sizes=(32), max_iter=50, random_state=0)
mlpclf.fit(X_train, y_train)
round(mlpclf.score(X_train, y_train), 3), round(mlpclf.score(X_test, y_test), 3)
mlpclf = MLPClassifier(hidden_layer_sizes=(32), max_iter=25, random_state=0)
mlpclf.fit(X_train, y_train)
round(mlpclf.score(X_train, y_train), 3), round(mlpclf.score(X_test, y_test), 3)
