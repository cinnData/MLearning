## [MLE-06] Validation examples ##

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

# Q1. Train-test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_train, y_train).round(3), reg.score(X_test, y_test).round(3)

# Q2. Repeat the process #
def check():
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	reg.fit(X_train, y_train)
	reg.score(X_train, y_train).round(3), reg.score(X_test, y_test).round(3)
check()
check()
check()

# Q3. Cross-validation #
from sklearn.model_selection import cross_val_score
val_scores = cross_val_score(reg, X, y, cv=3)
val_scores.round(3)
val_scores.mean().round(3)

# Q4. Using the mean absolute percentage error #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_train, y_pred_test = reg.predict(X_train), reg.predict(X_test)
from sklearn.metrics import mean_absolute_percentage_error as mape
mape(y_train, y_pred_train).round(3), mape(y_test, y_pred_test).round(3)
val_scores = cross_val_score(reg, X, y, cv=3,
    scoring='neg_mean_absolute_percentage_error')
val_scores.round(3)
