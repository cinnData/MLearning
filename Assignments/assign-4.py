## Assignment 5 ##

# Importing the data #
import numpy as np, pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'digits.csv.zip')

# Target vector and feature matrix #
y = df.iloc[:, 0]
y.value_counts()
X = df.iloc[:, 1:].values

# Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7)

# Q1. Random forest classifier #
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(max_depth=7, n_estimators=200)
rf1.fit(X_train, y_train)
round(rf1.score(X_train, y_train), 3), round(rf1.score(X_test, y_test), 3)
rf2 = RandomForestClassifier(max_depth=7, n_estimators=200, max_features=50)
rf2.fit(X_train, y_train)
round(rf2.score(X_train, y_train), 3), round(rf2.score(X_test, y_test), 3)
rf3 = RandomForestClassifier(max_depth=7, n_estimators=200, max_features=100)
rf3.fit(X_train, y_train)
round(rf3.score(X_train, y_train), 3), round(rf3.score(X_test, y_test), 3)

# Q2. XGBoost classifier #
import xgboost as xgb
xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, learning_rate=0.3)
xgb.fit(X_train, y_train)
round(xgb.score(X_train, y_train), 3), round(xgb.score(X_test, y_test), 3)

# Q3. Confusion matrix #
y_pred = xgb.predict(X_test)
conf = pd.crosstab(y_test, y_pred)
conf
total = conf.sum(axis=1)
right = np.diagonal(conf)
wrong = total - right
percent_wrong = 100*wrong/total
percent_wrong.round(1)

# Q4. Cluster analysis #
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=10, random_state=0)
clus.fit(X)
label = clus.labels_
crosstab = pd.crosstab(y, label)
crosstab
percent = 100*crosstab.max(axis=1)/crosstab.sum(axis=1)
percent.round(1)
