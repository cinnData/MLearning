## [MLE-11 ] Example - Airline passenger satisfaction ##

# Importing the data #
import pandas as pd, numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'airsat.csv')

# Exploring the data #
df.info()
df['sat'].mean().round(3)

# Target vector and feature matrix #
y = df['sat']
X = df.drop(columns='sat')

# Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Random forest model #
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
rf.fit(X_train, y_train)
rf.score(X_train, y_train).round(3), rf.score(X_test, y_test).round(3)

# XGBoost model #
from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=5, n_estimators=200, random_state=0)
xgb.fit(X_train, y_train)
xgb.score(X_train, y_train).round(3), xgb.score(X_test, y_test).round(3)

# Q3. Relevant features #
pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
pd.crosstab(df['business'], df['sat'])

# Q4. MLP model #
from tensorflow.keras import models, layers
network = [layers.Dense(32, activation='relu'), layers.Dense(2, activation='softmax')]
mlp = models.Sequential(network)
mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
mlp.fit(X_train, y_train, epochs=50, verbose=0);
round(mlp.evaluate(X_test, y_test, verbose=0)[1], 3)

# Q5. Multilayer perceptron model (normalized data) #
def normalize(x): 
    return (x - x.min())/(x.max() - x.min())
XN = X.apply(normalize)
XN_train, XN_test = train_test_split(XN, test_size=0.2, random_state=0)
mlp = models.Sequential(network)
mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
mlp.fit(XN_train, y_train, epochs=50, verbose=0);
round(mlp.evaluate(XN_test, y_test, verbose=0)[1], 3)

