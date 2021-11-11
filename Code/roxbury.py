## Example - Assessing home values in West Roxbury ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
fname = path + 'roxbury.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:5]
np.unique(data['floors'])

# Target vector and feature matrix #
y = data['value']
data.dtype.names
X = data[list(data.dtype.names[1:])]
from numpy.lib.recfunctions import structured_to_unstructured
X = structured_to_unstructured(X)
X.shape

# Linear regression equation #
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
round(linreg.score(X, y), 3)

# Predicted values #
ypred = linreg.predict(X)
r = np.corrcoef(y, ypred)[0,1]
round(r**2, 3)

# Scatter plot #
from matplotlib import pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(ypred, y, color='black', s=1)
plt.title('Figure 1. Actual vs predicted value')
plt.xlabel('Predicted value (thousands)')
plt.ylabel('Actual value (thousands)');

# Distribution of the home assessed value #
plt.figure(figsize=(8,6))
plt.title('Figure 2. Actual value')
plt.hist(y, color='gray', rwidth=0.97)
plt.xlabel('Actual value (thousands)');

# Trimmed data #
y_trim = y[(y >= 250) & (y <= 500)]
X_trim = X[(y >= 250) & (y <= 500)]
linreg.fit(X_trim, y_trim)
ypred_trim = linreg.predict(X_trim)
plt.figure(figsize=(6,6))
plt.scatter(ypred_trim, y_trim, color='black', s=1)
plt.title('Figure 3. Actual vs predicted value (trimmed data)')
plt.xlabel('Predicted value (thousands)')
plt.ylabel('Actual value (thousands)');

# Saving the model (edit path) #
import joblib
joblib.dump(linreg, 'linreg.pkl')
newlinreg = joblib.load('linreg.pkl')
np.sum(newlinreg.predict(X) != ypred)
