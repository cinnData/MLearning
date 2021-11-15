## Example - Handwritten digit recognition ##

# Importing the data #
import numpy  as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
d1 = np.genfromtxt(path + 'digits1.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d2 = np.genfromtxt(path + 'digits2.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d3 = np.genfromtxt(path + 'digits3.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d4 = np.genfromtxt(path + 'digits4.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d5 = np.genfromtxt(path + 'digits5.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d6 = np.genfromtxt(path + 'digits6.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
d7 = np.genfromtxt(path + 'digits7.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
data = np.concatenate([d1, d2, d3, d4, d5, d6, d7])
data.shape
np.unique(data['label'], return_counts=True)

# Target vector and feature matrix #
y = data['label']
from numpy.lib.recfunctions import structured_to_unstructured
X = structured_to_unstructured(data)[:, 1:]
np.unique(X)

# Plotting the images #
pic = X[0, :].reshape(28,28)
from matplotlib import pyplot as plt
plt.imshow(pic);
plt.imshow(pic, cmap='gray');
plt.gray()
plt.imshow(255 - pic);
pic = X[1, :].reshape(28,28)
plt.imshow(255 - pic);
pic = X[2, :].reshape(28,28)
plt.imshow(255 - pic);

# Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7)

# Decision tree classifier #
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_leaf_nodes=128)
treeclf.fit(X_train, y_train)
round(treeclf.score(X_train, y_train), 3), round(treeclf.score(X_test, y_test), 3)

# Random forest classifier #
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(max_leaf_nodes=128, n_estimators=10)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)
rfclf = RandomForestClassifier(max_leaf_nodes=128, n_estimators=50)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)
rfclf = RandomForestClassifier(max_leaf_nodes=128, n_estimators=100)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)
rfclf = RandomForestClassifier(max_depth=7, n_estimators=100)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)
rfclf = RandomForestClassifier(max_leaf_nodes=256, n_estimators=100)
rfclf.fit(X_train, y_train)
round(rfclf.score(X_train, y_train), 3), round(rfclf.score(X_test, y_test), 3)
