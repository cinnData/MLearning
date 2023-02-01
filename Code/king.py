## Example - House sales in King County ##

# Importing the data (edit path) #
import csv
with open('Dropbox/ml_course/data/king.csv', 'r') as conn:
	reader = csv.reader(conn)
	data = list(reader)
len(data)

# Headers #
header = data[0]
header
len(header)

# Target vector and feature matrix #
import numpy as np
Xy = np.array(data[1:])
Xy.shape
y = Xy[:, 15].astype(float)/1000
y.shape
X = Xy[:, 3:15].astype(float)
X.shape

# Q1. Distribution of the sale price #
from matplotlib import pyplot as plt
plt.figure(figsize=(8,6))
plt.title('Figure 1. Sale price')
plt.hist(y, color='gray', rwidth=0.97)
plt.xlabel('Sale price (thousands)');

# Q2. Linear regression model #
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
round(linreg.score(X, y), 3)
y_pred = linreg.predict(X)
np.corrcoef(y, y_pred)
np.corrcoef(y, y_pred)[0, 1]**2

# Q3. Actual price versus predicted price #
plt.figure(figsize=(6,6))
plt.scatter(x=y_pred, y=y, color='black', s=1)
plt.title('Figure 2. Actual vs predicted price')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)');
np.sum(y_pred < 0)
