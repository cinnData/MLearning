## Example - House sales in King County ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
fname = path + 'king.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:2]

# Target vector and feature matrix #
y = data['price']/1000
X = data[list(data.dtype.names[3:-1])]
from numpy.lib.recfunctions import structured_to_unstructured
X = structured_to_unstructured(X)
X.shape

# The distribution of the sale price #
from matplotlib import pyplot as plt
plt.figure(figsize=(8,6))
plt.title('Figure 1. Sale price')
plt.hist(y, color='gray', rwidth=0.97)
plt.xlabel('Sale price (thousands)');

# # Linear regression (first round) #
from sklearn.linear_model import LinearRegression
linreg1 = LinearRegression()
linreg1.fit(X, y)
round(linreg1.score(X, y), 3)
ypred1 = linreg1.predict(X)
plt.figure(figsize=(6,6))
plt.scatter(x=ypred1, y=y, color='black', s=1)
plt.title('Figure 2. Actual vs predicted price')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)');
np.sum(ypred1 < 0)

# Incorporating the zipcode #
X1 = data[list(data.dtype.names[5:-1])]
X1 = structured_to_unstructured(X1)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(data['zipcode'].reshape(21613,1))
X2 = enc.transform(data['zipcode'].reshape(21613,1)).toarray()
X2.shape
np.unique(X2, return_counts=True)
X = np.concatenate([X1, X2], axis=1)
X.shape

# Linear regression (second round) #
linreg2 = LinearRegression()
linreg2.fit(X, y)
round(linreg2.score(X, y), 3)
ypred2 = linreg2.predict(X)
plt.figure(figsize=(6,6))
plt.scatter(x=ypred2, y=y, color='black', s=1)
plt.title('Figure 3. Actual vs predicted price (2nd round)')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)');
np.sum(ypred2 < 0)






