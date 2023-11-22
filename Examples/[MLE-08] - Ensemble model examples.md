# [MLE-08] - Ensemble model examples

## Introduction

This exercise uses the data of the example MLE-02, which develops a linear regression model for predicting housing prices in King County. This model was validated in example MLE-06. Our objective, here, is to illustrate the obtention and evaluation of **ensemble regression models**. We use as a benchmark for these models, the linear regression model already discussed.

## Questions

Q1. Train and validate a **decision tree** regression model for predicting housing prices. Use the **mean absolute percentage error** (MAPE) as the performance metric for the model. 

Q2. The same for a **random forest** model built of 100 trees, with `max_depth=6`.

Q3. The same for a **gradient boosting** model.

Q4. Compare the distribution of the prediction errors in the test data between the linear regression model and the best of the models developed in questions Q1, Q2 and Q3. Examine the errors in dollar terms and in percentage terms.

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

We refresh the obtention of the linear regression model in example MLE-02 and validated in example MLE-06, which will be the benchmark for the ensemble models.

```
In [4]: from sklearn.linear_model import LinearRegression
   ...: lin = LinearRegression()
   ...: lin.fit(X_train, y_train)
   ...: lin.score(X_train, y_train).round(3), lin.score(X_test, y_test).round(3)
Out[4]: (0.786, 0.779)
```

So, a R-squared reference value could be 0.78. Though we can easily calculate the MAPE's, we use here the function `mean_absolute_percentage_error`, from the subpackage `metrics`.

```
In [5]: from sklearn.metrics import mean_absolute_percentage_error as mape
```

We get a reference of 20.6% for the MAPE:

```
In [6]: mape(y_train, lin.predict(X_train)).round(3), mape(y_test, lin.predict(X_test)).round(3)
Out[6]: (0.206, 0.206)
```

Finally, we refresh the scater plot of the actual and predicted prices, for complementary visual comparisons. Note that, since we use lin.predict(X) and y in `plt.scatter()`, we are plotting the prices for all the houses, though the model has been trained on a subset. If you are not cmfortable with this inconsistency, you can restrict the visualization to either training or test data.

```
In [7]: from matplotlib import pyplot as plt
   ...: plt.figure(figsize=(5,5))
   ...: plt.scatter(lin.predict(X), y, color='black', s=1)
   ...: plt.title('Figure 1. Linear regression')
   ...: plt.xlabel('Predicted value (thousands)')
   ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-08.1.png)

## Q1. Decision tree regression

To develop the decision tree model, we use the class `DecisionTreeRegressor()`, from the scikit-learn subpackage `tree`. We set the maximum depth as requested and train the model on one part of the data set.

```
In [8]: from sklearn.tree import DecisionTreeRegressor
   ...: tree = DecisionTreeRegressor(max_depth=6)
   ...: tree.fit(X_train, y_train)
Out[8]: DecisionTreeRegressor(max_depth=6)
```

We don't detect an overfitting problem here, but the R-squared value is too low.

```
In [9]: tree.score(X_train, y_train).round(3), tree.score(X_test, y_test).round(3)
Out[9]: (0.644, 0.65)
```

The MAPE confirms the previous comment.

```
In [10]: mape(y_train, tree.predict(X_train)).round(3), mape(y_test, tree.predict(X_test)).round(3)
Out[10]: (0.315, 0.314)
```

The scatter plot deserves two comments. First, the dots show a peculiar organization, stacked in "columns". This is explained by the way the decision tree makes the predictions, assigning the same predicted price to all the houses in the same leaf node. Second, there are no negative predicted prices, because the predicted price for every mode is the average price in that node, so the minimum possible predictive price is the minimum price in the data set, which is 75,000$. Decision tree models make conservative predictions, never out of the range of the data.

```
In [11]: plt.figure(figsize=(5,5))
    ...: plt.scatter(tree.predict(X), y, color='black', s=1)
    ...: plt.title('Figure 2. Decision tree regression')
    ...: plt.xlabel('Predicted value (thousands)')
    ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-08.2.png)

## Q2. Random forest regression

To develop the random forest model, we use the class `RandomForestRegressor()`, from the scikit-learn subpackage `tree`. 

```
In [12]: from sklearn.ensemble import RandomForestRegressor
```

We initialize an instance of this class setting `n_estimators=100`.

```
In [13]: rf = RandomForestRegressor(n_estimators=100, max_depth=6)
```

Now we follow the same steps as for the other models. The R-squared shows an improvement with respect to the decision tree, but is still below the value obtained with the linear regression model. Again, we don't detect overfitting.

```
In [14]: rf.fit(X_train, y_train)
    ...: rf.score(X_train, y_train).round(3), rf.score(X_test, y_test).round(3)
Out[14]: (0.7, 0.69)
```

The MAPE confirms the previous remarks, though the improvement looks more relevant than when compating R-squared values.

```
In [15]: mape(y_train, rf.predict(X_train)).round(3), mape(y_test, rf.predict(X_test)).round(3)
Out[15]: (0.305, 0.308)
```

Comparing the scatter plot with that of Figure 1, we see that the linear regression model gives better predictions for the cheap houses. Note that "wall" on the left side of the cloud of points. This is, as in Figure 2, a by-product of the prediction method.

```
In [16]: plt.figure(figsize=(5,5))
    ...: plt.scatter(rf.predict(X), y, color='black', s=1)
    ...: plt.title('Figure 3. Random forest regression')
    ...: plt.xlabel('Predicted value (thousands)')
    ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-08.3.png)

## Q3. Gradient boosting regression

We try now a gradient boosting regression model. We use the class `XGBRegress()`, from the package `xgboost`. The same can be done with the class `GradientBoostingRegressor()`, from the scikit-learn subpackage `tree`. Though `xgboost`is faster, no sensible difference would noticed in this case. `xgboost` is available in Google Colab. 

```
In [17]: from xgboost import XGBRegressor
```

We set the parameter as requested in question Q4. Note that the syntax is the same as in scikit-learn.

```
In [18]: xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
```

We find a clear case of overfitting, which is typical of gradient boosting. Neverthless, thios model is a bit better than the linear regression model on the test data.

```
In [19]: xgb.fit(X_train, y_train)
    ...: xgb.score(X_train, y_train).round(3), xgb.score(X_test, y_test).round(3)
Out[19]: (0.886, 0.802)
```
The MAPE also favors the gradient boosting model, though the results look a bit awkward, which is not rare for this type of data.

```
In [20]: mape(y_train, xgb.predict(X_train)).round(3), mape(y_test, xgb.predict(X_test)).round(3)
Out[20]: (0.176, 0.19)
```

The visualization also favors this new model. No negative predicted prices here, by the same reason as in the other two tree-based models. Also no "wall" effect on the left corner.

```
In [21]: plt.figure(figsize=(5,5))
    ...: plt.scatter(xgb.predict(X), y, color='black', s=1)
    ...: plt.title('Figure 4. Gradient boosting regression')
    ...: plt.xlabel('Predicted value (thousands)')
    ...: plt.ylabel('Actual value (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-08.4.png)

We are going to perform a more thorough examination of the two best models.

## Q5. Analysis of the prediction error

We calculate the errors for the two models, both in dollar and in percentage terms. We take only the test data

```
In [22]: lin_error = y_test - lin.predict(X_test)
    ...: lin_per_error = lin_error/y_test
    ...: xgb_error = y_test - xgb.predict(X_test)
    ...: xgb_per_error = xgb_error/y_test
```

We compare first the distributions of the raw errors (with sign). Overall, the XGBoost model has lower errors, meaning that the predicted prices are higher. The standard deviation is also a bit lower, which is good.

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

In percentage terms, the signed errors suggest similar remarks.

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

Dropping the sign, the erors in dollar terms have better statistics for the XGBoost model. 

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

Finally, absolute percentage errors also show better statistcs for the XGBoost model. The median, which is more informative than the mean here, is 2% lower, which is relevant. So, the detailed analysis of the prediction errors support the conclusion extracted from the performance metrics about the superiority of the XGBoost model.

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
