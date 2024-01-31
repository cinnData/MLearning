# [MLA-01] Assignment 1

## Introduction

This assignment is based on the example MLE-02, which develops a linear regression model for predicting **housing prices in King County**. Some shortcomings of the model obtained have already appeared in the discussion of the example:

* The model predicts negative prices for some houses.

* The distribution of the price is quite **skewed**, which is a fact of life in real state. So, the extreme values in the right tail of the distribution can exert an undesired influence on the coefficients of the regression equation. Transformations, such as the square root or the logarithm, are recommended in Statistics textbooks in many situations. In particular, the **log transformation** is recommended for variables with skewed distributions, to limit the influence of extreme values.

* The strong **correlation** between actual and predicted prices may be an artifact created by those extreme values.

* It could be argued that a model based on a linear equation does not make sense on such a wide range of prices. In order to cope with the high prices of the supermansions, we may be spoiling the prediction for the nonexpensive houses.

* The linear regression equation is obtained by means of the **least squares method**, which is based on minimizing the sum of squares of the prediction errors. But this gives the same weight to an error of $30,000 in predicting a price of $100,000 as in predicting one of $7,000,000, which does not make business sense. Maybe we should take a look at the errors in percentage terms.

## Questions

Q1. The role of longitude and latitude in the prediction of real estate prices is unclear. Do they really contribute to get better predictions in the first model of this example? If we keep them in the second model, do we get a better model? 

Q2. Evaluate in dollar terms the predictive performance of the two models presented in this example. For instance, you can use the mean (or median) absolute error. Can you make a statement like "the value of *x*% of the houses can be predicted with an error below *y* thousand dollars"?

Q3. Is it better to use the percentage error in the above assessment?

Q4. Can the strong correlation be an artifact created by the extreme values? Trim the data set, dropping the houses beyond a certain threshold of price and/or size. Do you get a better model?

Q5. The distribution of the price is quite skewed, which is a fact of life in real state. The extreme values in the right tail of the distribution can exert an undesired influence on the regression coefficients. Develop and evaluate a model for predicting the price that is based on a linear regression equation which has the logarithm of the price on the left side. 

## Submission

1. Submit, through Blackboard, a readable and printable report responding these questions and explaining what you have done, including Python input and output. This can be a Word document, a PDF document or a Jupyter notebook (.ipynb).

2. Put your name on top of the document.

## Deadline

February 4 (Sunday), 24:00.
