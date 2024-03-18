## Assignment 5 ##

# Importing the data #
import numpy as np, pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df1 = pd.read_csv(path + 'zalando1.csv.zip')
df2 = pd.read_csv(path + 'zalando2.csv.zip')
df = pd.concat([df1, df2,])

# Target vector and feature matrix #
y = df.iloc[:, 0]
y.value_counts()
X = df.iloc[:, 1:].values/255

# Plot one image of every type #
from matplotlib import pyplot as plt
plt.gray()
X0 = X[y == 0, :]
plt.imshow(1 - X0[0, :].reshape(28,28));
X1 = X[y == 1, :]
plt.imshow(1 - X1[0, :].reshape(28,28));
X2 = X[y == 2, :]
plt.imshow(1 - X2[0, :].reshape(28,28));
X3 = X[y == 3, :]
plt.imshow(1 - X3[0, :].reshape(28,28));
X4 = X[y == 4, :]
plt.imshow(1 - X4[0, :].reshape(28,28));
X5 = X[y == 5, :]
plt.imshow(1 - X5[0, :].reshape(28,28));
X6 = X[y == 6, :]
plt.imshow(1 - X6[0, :].reshape(28,28));
X7 = X[y == 7, :]
plt.imshow(1 - X7[0, :].reshape(28,28));
X8 = X[y == 8, :]
plt.imshow(1 - X8[0, :].reshape(28,28));
X9 = X[y == 9, :]
plt.imshow(1 - X9[0, :].reshape(28,28));

# Q2. Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)

# Q3. Ensemble classifier #
import xgboost as xgb
clf1 = xgb.XGBClassifier(max_depth=7, n_estimators=200, learning_rate=0.3)
clf1.fit(X_train, y_train)
round(clf1.score(X_train, y_train), 3), round(clf1.score(X_test, y_test), 3)

# Q4a. MLP classifier #
from tensorflow.keras import models, layers
mlp_net = [layers.Dense(32, activation='relu'), layers.Dense(10, activation='softmax')]
clf2 = models.Sequential(layers=mlp_net)
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

# Q4b. CNN classifier #
Z_train, Z_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
cnn_net = [layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')]
clf3 = models.Sequential(cnn_net)
clf3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf3.fit(Z_train, y_train, epochs=10, validation_data=(Z_test, y_test));

# Q5a. Clothes better predicted by the XGB model #
y_pred1 = clf1.predict(X_test)
pd.crosstab(y_test, y_pred1)

# Q5b. Clothes better predicted by the MLP model #
y_pred2 = clf2.predict(X_test).argmax(axis=1)
pd.crosstab(y_test, y_pred2)

# Q5c. Clothes better predicted by the CNN model #
y_pred3 = clf3.predict(Z_test).argmax(axis=1)
pd.crosstab(y_test, y_pred3)
