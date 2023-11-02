# [ML-07E1] Validation examples

## Introduction

This validation exercise uses the data of the example **House sales in King County**, which develops a linear regression model for predicting housing prices. Assuming that the evaluation of the model is based on the R2 statistic, as returned by the scikit-learn method `.score()`, we follow standard procedures.

## Questions

The questions are:

Q1. Perform a train-test split of the King County data to explore whether there is an overfitting issue for the model developed in the example.

Q2. Repeat the process several times, to see how the randomness of the split affects the figures obtained.

Q3. Use a cross-validation approach for the same purpose.

## Importing the King County data

We import the data from the GitHub repository, as we did previously. We use the function `read_csv()` with an index specification. 

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'king.csv', index_col=0)
```

We also prepare the target vector and the feature matrix as we did then.

```
In [2]: y = df.iloc[:, -1]
   ...: X1 = df.iloc[:, 4:-1]
   ...: X2 = pd.get_dummies(df['zipcode'])
   ...: X = pd.concat([X1, X2], axis=1)
   ...: X = X.values
```

## Q1. Train-test split

We import the function `train_test_split()`, from the scikit-learn subpackage `sklearn.model_selection`.

```
In [3]: from sklearn.model_selection import train_test_split
```

We apply now this function to the pair `X`, `y`, specifying that a 20% of the data units will go to the test set. We two arrays (train and test) for every input.

```
In [4]: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

The function `train_test_split()` takes a collection of positional arguments such as lists, NumPy arrays or Pandas data containers, of the same length, splitting them at random, according to the proportion specified by the parameter `test_size` (alternatively, you can specify `train_size`). The split is the same for all the inputs, which is what we need for the model valiodation.

Indeed, we can check the shapes of the train and test parts:

```
In [5]: X_train.shape, y_train.shape
Out[5]: ((17290, 80), (17290,))
```
```
In [6]: X_test.shape, y_test.shape
Out[6]: ((4323, 80), (4323,))
```

We import now the class `LinearRegression()`, creating an instance that we call `reg`.  

```
In [7]: from sklearn.linear_model import LinearRegression
   ...: reg = LinearRegression()
```
 The estimator `reg` is fitted to the *training data*. This will calculate the regression coefficients that are optimal (minimum MSE) for the training data.

```
In [8]: reg.fit(X_train, y_train)
Out[8]: LinearRegression()
```

Now the performance of the equation obtained is evaluated in both the training and the test data.

```
In [9]: reg.score(X_train, y_train).round(3), reg.score(X_test, y_test).round(3)
Out[9]: (0.788, 0.773)
```

The R2 value obtained on the training data is 1.5% higher than the value obtianed on the test data. Most practitioners would say that these figures do not constitute an "overfitting issue". Since there is always a residual of subjectivity in such an assertion, it would be wiser to use the test value to evaluatre the model.

# Q2. Repeat the process

As it is provided by the function `train_test_split()`, the train-test split is just a random selection of data units. So, additional runs will produce different splits and, hence, different R2 values. To explore how this could affect our results, we define a function which integrates splitting the data, fitting an equation and evaluating it.

```
In [10]: def check():
    ...: 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ...: 	reg.fit(X_train, y_train)
    ...: 	return reg.score(X_train, y_train).round(3), reg.score(X_test, y_test).round(3)
```

Now, we apply this function three times:

```
In [11]: check()
Out[11]: (0.786, 0.782)
```

```
In [12]: check()
Out[12]: (0.787, 0.779)
```
```
In [13]: check()
Out[13]: (0.786, 0.781)
```

After these checks, we feel more comfortable with our validation process.

## Q3. Cross-validation

For the cross-validation experience, we import the function `cross_val_score()`, from the scikit-learn subpackage `model_selection`.

```
In [14]: from sklearn.model_selection import cross_val_score
```

We apply this function to the estimator `reg`, the target vector `y` and the feature matrix `X`. The preceding splits and fits do not affect the cross-validation. Here, since we specify `cv=3`, the data will be randomly split in three parts of the same size, and three fits will be performed, training on two thirds of the data and testing on other third. The values returned are the R2 statistics calculated on the test part.

```
In [15]: val_scores = cross_val_score(reg, X, y, cv=3)
    ...: val_scores.round(3)
Out[15]: array([0.78 , 0.769, 0.794]))
```

A practitioner may use here the average of these R2 values for scoring the model.

```
In [16]: val_scores.mean().round(3)
Out[16]: 0.781
```
