# 6. Linear regression

### Introduction

In machine learning, the term **regression** applies to the prediction of a numeric target. Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind when we think about "predicting".


When the equation is linear, as in

<img src="https://render.githubusercontent.com/render/math?math=\large Y = b_0 %2B b_1 X_1 %2B b_2 X_2 %2B \cdots %2B b_k X_k,">

we have **linear regression**, which is the subject of this chapter. The predictions of a linear regression model can typically be improved by more sophisticated techniques, but most analysts start there, because it helps them to understand the data. An alternative approach, based on **decision trees**, will be discussed later in this course.

### Evaluation of a linear regression model

In general, regression models are evaluated through their prediction errors. The basic schema is

<img src="https://render.githubusercontent.com/render/math?math=\large \textrm{Prediction\ error} = \textrm{Actual\ value} - \textrm{Predicted\ value}.">

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the sum of the squared residuals is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in other regression models.

Statisticians look at the **residual sum of squares** for evidence of good fit between the regression model and the data. The *R***-squared statistic** is a standardized measure which operationalizes this. More specifically, they take advantage of the variance decomposition formula

<img src="https://render.githubusercontent.com/render/math?math=\large R^2 = \displaystyle \frac{\textrm{var(Predicted\ values)}} {\textrm{var(Actual\ values)}}\,.">

It turns out that the square root of this proportion coincides with the correlation between the actual and the predicted values, called the **multiple correlation** in statistics textbooks. Although this is no longer true for other regression methods, this correlation is still the standard approach to the evaluation of a regression model.

### Linear regression in scikit-learn

The scikit-learn subpackage `linear_model` provides various regression estimators. The beginner's choice is the class `LinearRegression`, with its default parameters (meaning that you leave the parenthesis empty). You can instantiate an estimator from this class as:

`from sklearn.linear_model import LinearRegression`

`linreg = LinearRegression()`

Note that `linreg` is a user-provided name. `LinearRegression` comes with the three basic methods, `fit`, `predict` and `score`, already mentioned in this course. As for any regression estimator, `score` returns a *R*-squared value.
