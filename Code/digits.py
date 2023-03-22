## Example - Handwritten digit recognition ##

# Importing the data #
import csv, numpy as np
with open('Dropbox/ml_course/data/digits.csv', mode='r') as conn:
	reader = csv.reader(conn)
	data = list(reader)
len(data)

# Headers #
header = data[0]
header[:10]
len(header)

# Target vector and feature matrix #
Xy = np.array(data[1:])
Xy.shape
y = Xy[:, 0].astype(float)
y.shape
X = Xy[:, 1:].astype(float)
X.shape

# Exploring the data #
np.unique(y, return_counts=True)
np.unique(X)

# Q1. Plotting the first image #
pic = X[0, :].reshape(28,28)
from matplotlib import pyplot as plt
plt.imshow(pic);
plt.imshow(pic, cmap='gray');
plt.gray()
plt.imshow(255 - pic);

# Q2. Plotting other images #
pic = X[1, :].reshape(28,28)
plt.imshow(255 - pic);
pic = X[2, :].reshape(28,28)
plt.imshow(255 - pic);

# Q3. Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7)

# Q4. Decision tree classifier #
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_leaf_nodes=128)
treeclf.fit(X_train, y_train)
round(treeclf.score(X_train, y_train), 3), round(treeclf.score(X_test, y_test), 3)

# Q5. Random forest classifier #
from sklearn.ensemble import RandomForestClassifier
rfclf1 = RandomForestClassifier(max_leaf_nodes=128, n_estimators=10)
rfclf1.fit(X_train, y_train)
round(rfclf1.score(X_train, y_train), 3), round(rfclf1.score(X_test, y_test), 3)

# Q6. Change the specification #
rfclf2 = RandomForestClassifier(max_leaf_nodes=128, n_estimators=50)
rfclf2.fit(X_train, y_train)
round(rfclf2.score(X_train, y_train), 3), round(rfclf2.score(X_test, y_test), 3)
rfclf3 = RandomForestClassifier(max_leaf_nodes=128, n_estimators=100)
rfclf3.fit(X_train, y_train)
round(rfclf3.score(X_train, y_train), 3), round(rfclf3.score(X_test, y_test), 3)
rfclf4 = RandomForestClassifier(max_depth=7, n_estimators=100)
rfclf4.fit(X_train, y_train)
round(rfclf4.score(X_train, y_train), 3), round(rfclf4.score(X_test, y_test), 3)
rfclf5 = RandomForestClassifier(max_leaf_nodes=256, n_estimators=100)
rfclf5.fit(X_train, y_train)
round(rfclf5.score(X_train, y_train), 3), round(rfclf5.score(X_test, y_test), 3)
