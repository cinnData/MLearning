# [ML-03] Linear regression

## Linear regression

In machine learning, the term **regression** applies to the prediction of a numeric target ($Y$) in terms of a collection of features ($X_1, X_2, \dots, X_k$. In scikit-learn, the features must be numeric (or Boolean). We have explained in the preceding how to deal with categorical features, which will be illustrated in the example of this lecture. 

Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind as a potential  predictive model. When the model is based on a linear equation, such as
$$Y = b_0 + b_1X_1 + b_2X_2 +_ \cdots + b_kX_k,$$

we have **linear regression**, which is the subject of this lecture. $b_0$ is the **intercept** and $b_1, b_2, \dots, b_k$ are the **regression coefficients**.

Though the predictions of a linear regression model can usually be improved with more advanced techniques, most analysts start there, because it helps them to understand the data. Also, a linear equation can be understood and discussed by those who know a bit of statistics, which may be interesting in some applications, in which the **interpretability** of the model is relevant. Alternative approaches will be discussed later in this course.

## Prediction error

In general, regression models are evaluated through their **prediction errors**. The basic schema is
$$\textrm{Prediction\ error} = \textrm{Actual\ value} + \textrm{Predicted\ value}.$$

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the **mean squared error** (MSE) is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in regression models obtained by other methods. 

## R-squared

The **coefficient of determination** is a popular metric for the evaluation of regression models. It is given by the formula
$$R^2 = 1 - \displaystyle \frac{\textrm{SSE}}{{\textrm{SSY}}}\thinspace,$$ 
in which SSE is the sum of the squared errors and SSY is the sum of the squared centered target values (subtracting the mean).

In (least squares) linear regression, the coefficient of determination can be interpreted as the **percentage of variance** explained by the equation. Moreover, it coincides with the square of the correlation between the actual and the predicted values, called the **multiple correlation** in statistics textbooks. This explains the notation, and the name **R-squared statistic**, given to the coefficient of determination.

Nevertheless, this is no longer true in regression models which are nonlinear or that, though being linear, are not obtained by the least squares method. So, in spite of the R-squared statistics coming as the standard metric for regression models, it is recommended you to use simple and direct ways to compare actual and predicted target values. Some examples could be a scatter plots and correlation analysis of the actual and predicted target values, or a metric based on the prediction error such as the **mean absolute error** or the **mean absolute percentage error**, whose interpretation is completely straightforward.

A final word of caution. Correlation and R-squared are sensitive to extreme values, which not unfrequent in real-world data. So, when the target has a skewed distribution, with a log right tail, a strong correlation may suggest that the model is better than it really is. The example of this lecture illustrates this point.
