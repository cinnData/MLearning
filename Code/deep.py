## [MLE-12] Deep learning examples ##

# Importing the data #
import numpy as np, pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'digits.csv.zip')

# Target vector and feature matrix #
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Q1. Train-test split #
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1/7, random_state=0)

# Q2. MLP model #
from tensorflow.keras import models, layers
net1 = [layers.Dense(32, activation='relu'), layers.Dense(10, activation='softmax')]
clf1 = models.Sequential(layers=net1)
clf1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf1.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));
clf1.summary()

# Q3. Prediction with a MLP model #
clf1.predict(X_test[:1, :])
y_test[0]

# Q4. Rescaling the data #
X = X/255
X_train, X_test = model_selection.train_test_split(X, test_size=1/7, random_state=0)
clf2 = models.Sequential(net1)
clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));

# Q5. Convolutional neural network #
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
net2 = [layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')]
clf3 = models.Sequential(net2)
clf3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf3.fit(Z_train, y_train, epochs=10, validation_data=(Z_test, y_test));
clf3.summary()

