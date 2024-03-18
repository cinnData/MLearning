## Transfer learning ##

# Resources #
import numpy as np, pandas as pd
from sklearn import model_selection
from tensorflow.keras import models, layers

# Importing the data [1] #
import numpy as np, pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'digits.csv.zip')
y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values/255
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1/7, random_state=0)

# Convolutional neural network #
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
network = [layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')]
clf = models.Sequential(network)
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));

# Importing the data [2] #
df1 = pd.read_csv(path + 'zalando1.csv.zip')
df2 = pd.read_csv(path + 'zalando2.csv.zip')
df = pd.concat([df1, df2,])
y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values/255
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1/7, random_state=0)

# Convolutional neural network #
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test));

