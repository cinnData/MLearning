## [MLA-01] Assignment 1 ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'king.csv', index_col=0)
df['price'] = df['price']/1000

# Model 1 #
y = df.iloc[:, -1]
X = df.iloc[:, 2:-1]
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()
reg1.fit(X, y)
reg1.score(X, y).round(3)
y_pred1 = reg1.predict(X)

# Model 2 #
X1 = df.iloc[:, 4:-1]
X2 = pd.get_dummies(df['zipcode'])
X = pd.concat([X1, X2], axis=1).values
reg2 = LinearRegression()
reg2.fit(X, y)
reg2.score(X, y).round(3)
y_pred2 = reg2.predict(X)

# Q1. Role of longitude and latitude in the prediction #
X_min = df.iloc[:, 4:-1]
reg3 = LinearRegression()
reg3.fit(X_min, y)
reg3.score(X_min, y).round(3)
X_max = pd.concat([df.iloc[:, 2:-1], X2], axis=1).values
reg4 = LinearRegression()
reg4.fit(X_max, y)
reg4.score(X_max, y).round(3)

# Q2. Evaluate in dollar terms the predictive performance #
err1 = y - y_pred1 
abs(err1).describe().round(3)
err2 = y - y_pred2 
abs(err2).describe().round(3)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, y_pred1).round(3), mean_absolute_error(y, y_pred2).round(3)

# Q3. Evaluate in percentage terms #
(abs(err1)/y).describe().round(3)
(abs(err2)/y).describe().round(3)
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y, y_pred1).round(3), mean_absolute_percentage_error(y, y_pred2).round(3)

# Q4. Trim the data set #
y_trim = y[y.between(100, 1000)]
X_trim = X[y.between(100, 1000)]
reg4 = LinearRegression()
reg4.fit(X_trim, y_trim)
reg4.score(X_trim, y_trim).round(3)
y_pred4 = reg4.predict(X_trim)
mean_absolute_percentage_error(y_trim, y_pred4).round(3)

# Q5. Logarithmic transformation #
import numpy as np
reg2.fit(X, np.log(y))
reg2.score(X, np.log(y))
y_pred = np.exp(reg2.predict(X))
mean_absolute_percentage_error(y, y_pred)
