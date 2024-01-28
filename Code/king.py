## [MLE-02] Example - House sales in King County ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'king.csv', index_col=0)

# Exploring the data #
df.info()
df.index.duplicated().sum()
df.duplicated().sum()
duplicates = df.index[df.index.duplicated()]
df.loc[duplicates].head()
df['price'] = df['price']/1000

# Q1. Distribution of the sale price #
df['price'].describe()
from matplotlib import pyplot as plt
plt.figure(figsize=(7,5))
plt.title('Figure 1. Actual price')
plt.hist(df['price'], color='gray', edgecolor='white')
plt.xlabel('Sale price (thousands)');

# Q2. Linear regression equation #
y = df.iloc[:, -1]
X = df.iloc[:, 2:-1]
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
reg.score(X, y).round(3)

# Q3. Plot the actual price versus the price predicted by your model #
plt.figure(figsize=(5,5))
plt.title('Figure 2. Actual price vs predicted price')
plt.scatter(x=y_pred, y=y, color='black', s=1)
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)');
plt.figure(figsize=(5,5))
plt.title('Figure 3. Absolute prediction error vs predicted price')
plt.scatter(x=y_pred, y=abs(y-y_pred), color='black', s=1)
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Absolute predicted error (thousands)');
(y_pred < 0).sum()

# Q4. Dummies for the zipcodes #
X1 = df.iloc[:, 4:-1]
X2 = pd.get_dummies(df['zipcode'])
X2.head()
X = pd.concat([X1, X2], axis=1)
X.shape
X = X.values
reg.fit(X, y)
y_pred = reg.predict(X)
reg.score(X, y).round(3)
plt.figure(figsize=(5,5))
plt.title('Figure 4. Actual price vs predicted price')
plt.scatter(x=y_pred, y=y, color='black', s=1)
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)');
(y_pred < 0).sum()
