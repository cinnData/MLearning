## [MLE-08] - Ensemble model examples ##

# Importing the King County data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'king.csv', index_col=0)
df['price'] = df['price']/10**3
y = df.iloc[:, -1]
X1 = df.iloc[:, 4:-1]
X2 = pd.get_dummies(df['zipcode'])
X = pd.concat([X1, X2], axis=1)
X = X.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Q1. Linear regression #
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train, y_train)
lin.score(X_train, y_train).round(3), lin.score(X_test, y_test).round(3)
from sklearn.metrics import mean_absolute_percentage_error as mape
mape(y_train, lin.predict(X_train)).round(3), mape(y_test, lin.predict(X_test)).round(3)
# Plotting #
from matplotlib import pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(lin.predict(X), y, color='black', s=1)
plt.title('Figure 1. Linear regression')
plt.xlabel('Predicted value (thousands)')
plt.ylabel('Actual value (thousands)');

# Q2. Decision tree regression #
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=6)
tree.fit(X_train, y_train)
tree.score(X_train, y_train).round(3), tree.score(X_test, y_test).round(3)
mape(y_train, tree.predict(X_train)).round(3), mape(y_test, tree.predict(X_test)).round(3)
# Plotting #
plt.figure(figsize=(5,5))
plt.scatter(tree.predict(X), y, color='black', s=1)
plt.title('Figure 2. Decision tree regression')
plt.xlabel('Predicted value (thousands)')
plt.ylabel('Actual value (thousands)');

# Q3. Random forest regression #
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=6)
rf.fit(X_train, y_train)
rf.score(X_train, y_train).round(3), rf.score(X_test, y_test).round(3)
mape(y_train, rf.predict(X_train)).round(3), mape(y_test, rf.predict(X_test)).round(3)
# Plotting #
plt.figure(figsize=(5,5))
plt.scatter(rf.predict(X), y, color='black', s=1)
plt.title('Figure 3. Random forest regression')
plt.xlabel('Predicted value (thousands)')
plt.ylabel('Actual value (thousands)');

# Q4. Gradient boosting regression ##
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb.fit(X_train, y_train)
xgb.score(X_train, y_train).round(3), xgb.score(X_test, y_test).round(3)
mape(y_train, xgb.predict(X_train)).round(3), mape(y_test, xgb.predict(X_test)).round(3)
# Plotting #
plt.figure(figsize=(5,5))
plt.scatter(xgb.predict(X), y, color='black', s=1)
plt.title('Figure 4. Gradient boosting regression')
plt.xlabel('Predicted value (thousands)')
plt.ylabel('Actual value (thousands)');

# Q5. Analysis of the prediction error #
lin_error = y_test - lin.predict(X_test)
lin_per_error = lin_error/y_test
xgb_error = y_test - xgb.predict(X_test)
xgb_per_error = xgb_error/y_test
pd.concat([lin_error.describe(), xgb_error.describe()], axis=1)
pd.concat([lin_per_error.describe(), xgb_per_error.describe()], axis=1)
pd.concat([lin_error.abs().describe(), xgb_error.abs().describe()], axis=1)
pd.concat([lin_per_error.abs().describe(), xgb_per_error.abs().describe()], axis=1)
