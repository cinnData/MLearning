# [ML-03] Linear regression

## Linear regression

In machine learning, the term **regression** applies to the prediction of a numeric target. Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind as a potential  predictive model.

When the model is based on a linear equation, as in
$$Y = b_0 + b_1X_1 + b_2X_2 +_ \cdots + b_kX_k,$$

we have **linear regression**, which is the subject of this lecture. Though the predictions of a linear regression model can usually be improved with more advanced techniques, most analysts start there, because it helps them to understand the data. An alternative approach, based on **decision trees**, will be discussed later in this course.

## Evaluation of a linear regression model

In general, regression models are evaluated through their **prediction errors**. The basic schema is
$$\textrm{Prediction\ error} = \textrm{Actual\ value} + \textrm{Predicted\ value}.$$

The **coefficient of determination** is a popular metric for the evaluation of regression models. It is given by the formula
$$R^2 = 1 - \displaystyle \frac{\textrm{SSE}}{{\textrm{SSY}}}\thinspace,$$ 
in which SSE is the sum of the squared errors and SSY is the sum of the squared centered target values (subtracting the mean).

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the **mean squared error** (MSE) is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in regression models obtained by other methods. In that case, the coefficient of determination can be interpreted as the **percentage of variance** explained by the equation. Moreover, it coincides with the square of the correlation between the actual and the predicted values, called the **multiple correlation** in statistics textbooks. This explains the notation, and the name **R-squared statistic**, given to the coefficient of determination.

Nevertheless, this is no longer true in regression models which are nonlinear or that, though being linear, are not obtained by the least squares method. So, in spite of the R-squared statistics coming as the standard metric for regression models, I would recommend you to use the correlation of actual and predicted values or a metric based on the prediction error such as the **mean absolute error** or the **mean absolute percentage error**, whose interpretation is completely straightforward.
