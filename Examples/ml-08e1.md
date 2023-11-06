# [ML-E8E1] - Ensemble model examples

## Introduction

This exercise uses the data of the example **House sales in King County**, which develops a linear regression model for predicting housing prices. The objective is to illustrate the obtention and evaluation of ensemble regression models, comparing them to a plain linear regression model.

## Questions

The questions are:

## Importing the King County data

We import the data from the GitHub repository, as we did previously.  

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'king.csv', index_col=0)
   ...: df['price'] = df['price']/10**3
```

We prepare the target vector and the feature matrix as we did then.

```
In [2]: y = df.iloc[:, -1]
   ...: X1 = df.iloc[:, 4:-1]
   ...: X2 = pd.get_dummies(df['zipcode'])
   ...: X = pd.concat([X1, X2], axis=1)
   ...: X = X.values
```

Finally, we perform a train-test split for the validation of the models obtained in this exercise.

```
In [3]: from sklearn.model_selection import train_test_split
   ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## Q1. Linear regression

```
In [4]: from sklearn.linear_model import LinearRegression
   ...: lin = LinearRegression()
   ...: lin.fit(X_train, y_train)
   ...: lin.score(X_train, y_train).round(3), lin.score(X_test, y_test).round(3)
Out[4]: (0.786, 0.779)
```

```
In [5]: from sklearn.metrics import mean_absolute_percentage_error as mape
```

```
In [6]: mape(y_train, lin.predict(X_train)).round(3), mape(y_test, lin.predict(X_test)).round(3)
Out[6]: (0.206, 0.206)
```

```
In [7]: from matplotlib import pyplot as plt
   ...: plt.figure(figsize=(5,5))
   ...: plt.scatter(lin.predict(X), y, color='black', s=1)
   ...: plt.title('Figure 1. Linear regression')
   ...: plt.xlabel('Predicted value (thousands)')
   ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e1.1.png)

## Q2. Decision tree regression

```
In [8]: from sklearn.tree import DecisionTreeRegressor
   ...: tree = DecisionTreeRegressor(max_depth=6)
   ...: tree.fit(X_train, y_train)
Out[8]: DecisionTreeRegressor(max_depth=6)
```

```
In [9]: tree.score(X_train, y_train).round(3), tree.score(X_test, y_test).round(3)
Out[9]: (0.644, 0.65)
```

```
In [10]: mape(y_train, tree.predict(X_train)).round(3), mape(y_test, tree.predict(X_test)).round(3)
Out[10]: (0.315, 0.314)
```

```
In [11]: plt.figure(figsize=(5,5))
    ...: plt.scatter(tree.predict(X), y, color='black', s=1)
    ...: plt.title('Figure 2. Decision tree regression')
    ...: plt.xlabel('Predicted value (thousands)')
    ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e1.2.png)

## Q3. Random forest regression

```
In [12]: from sklearn.ensemble import RandomForestRegressor
```

```
In [13]: rf = RandomForestRegressor(n_estimators=100, max_depth=6)
```

```
In [14]: rf.fit(X_train, y_train)
    ...: rf.score(X_train, y_train).round(3), rf.score(X_test, y_test).round(3)
Out[14]: (0.7, 0.69)
```

```
In [15]: mape(y_train, rf.predict(X_train)).round(3), mape(y_test, rf.predict(X_test)).round(3)
Out[15]: (0.305, 0.308)
```

```
In [16]: plt.figure(figsize=(5,5))
    ...: plt.scatter(rf.predict(X), y, color='black', s=1)
    ...: plt.title('Figure 3. Random forest regression')
    ...: plt.xlabel('Predicted value (thousands)')
    ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e1.3.png)

## Q4. Gradient boosting regression

```
In [17]: from xgboost import XGBRegressor
```

```
In [18]: xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
```

```
In [19]: xgb.fit(X_train, y_train)
    ...: xgb.score(X_train, y_train).round(3), xgb.score(X_test, y_test).round(3)
Out[19]: (0.886, 0.802)
```

```
In [20]: mape(y_train, xgb.predict(X_train)).round(3), mape(y_test, xgb.predict(X_test)).round(3)
Out[20]: (0.176, 0.19)
```

```
In [21]: plt.figure(figsize=(5,5))
    ...: plt.scatter(xgb.predict(X), y, color='black', s=1)
    ...: plt.title('Figure 4. Gradient boosting regression')
    ...: plt.xlabel('Predicted value (thousands)')
    ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e1.4.png)

## Q5. Analysis of the prediction error

```
In [22]: lin_error = y_test - lin.predict(X_test)
    ...: lin_per_error = lin_error/y_test
    ...: xgb_error = y_test - xgb.predict(X_test)
    ...: xgb_per_error = xgb_error/y_test
```

```
In [23]: pd.concat([lin_error.describe(), xgb_error.describe()], axis=1)
Out[23]: 
             price        price
count  4323.000000  4323.000000
mean      3.249821    -0.304062
std     180.779721   171.390355
min   -1047.832784 -2469.403564
25%     -72.535336   -71.403687
50%       0.988696   -13.795135
75%      65.708204    48.901962
max    4284.892274  4314.262939
```

```
In [24]: pd.concat([lin_per_error.describe(), xgb_per_error.describe()], axis=1)
Out[24]: 
             price        price
count  4323.000000  4323.000000
mean     -0.013164    -0.076317
std       0.285027     0.271567
min      -3.192581    -2.743782
25%      -0.157108    -0.194391
50%       0.002170    -0.030666
75%       0.145195     0.088940
max       1.972441     0.695424
```

```
In [25]: pd.concat([lin_error.abs().describe(), xgb_error.abs().describe()], axis=1)
Out[25]: 
             price        price
count  4323.000000  4323.000000
mean    102.560052    93.965716
std     148.898869   143.328809
min       0.002928     0.011169
25%      32.862605    27.368179
50%      68.683439    61.153809
75%     123.048275   114.631226
max    4284.892274  4314.262939
```

```
In [26]: pd.concat([lin_per_error.abs().describe(), xgb_per_error.abs().describe()], axis=1)
Out[26]: 
             price        price
count  4323.000000  4323.000000
mean      0.206127     0.190137
std       0.197271     0.208360
min       0.000005     0.000021
25%       0.069912     0.060316
50%       0.151103     0.132062
75%       0.284038     0.249484
max       3.192581     2.743782
```
