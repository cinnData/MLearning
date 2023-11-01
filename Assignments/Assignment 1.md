# Assignment 1

## Introduction

This assignment is based on the example **House sales in King County**, which develops a linear regression model for predicting housing prices. Some shortcomings of the model obtained have already appeared in the discussion of the example:

* The model predicts negative prices for some houses.

* The distribution of the price is quite **skewed**, which is a fact of life in real state. So, the extreme values in the right tail of the distribution can exert an undesired influence on the coefficients of the regression equation. Transformations, such as the square root or the logarithm, are recommended in Statistics textbooks in many situations. In particular, the **log transformation** is recommended for variables with skewed distributions, to limit the influence of extreme values.

* The strong **correlation** between actual and predicted prices may be an artifact created by those extreme values.

* It could be argued that a model based on a linear equation does not make sense on such a wide range of prices. In order to cope with the high prices of the supermansions, we may be spoiling the prediction for the nonexpensive houses.

* The linear regression equation is obtained by means of the **least squares method**, which is based on minimizing the sum of squares of the prediction errors. But this gives the same weight to an error of $30,000 in predicting a price of $100,000 as in predicting one of $7,000,000, which does not make business sense. Maybe we should take a look at the errors in percentage terms.

## Questions

Q1. Evaluate in dollar terms the predictive performance of the model presented in this example. A statement in this line could be something like "the sale price of x% of the houses can be predicted with an error below y thousand dollars".

Q2. Would it be better to evaluate the model expressing the prediction error in percentage terms? Then our statement would be like "the price of x% of the houses can be predicted with an error below y% of the actual price".

Q3. Develop a model for predicting the sale price which is based on a linear regression equation which has the logarithm of the price on the left side. Do you get better predictions with this model? Does this respond to the objections listed above?
