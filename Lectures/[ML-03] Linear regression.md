# [ML-03] Linear regression

## Linear regression

In machine learning, the term **regression** applies to the prediction of a numeric target ($Y$) in terms of a collection of features ($X_1, X_2, \dots, X_k$. In scikit-learn, the features must be numeric (or Boolean). We have explained in the preceding lecture (ML-02)how to deal with categorical features, which will be illustrated in the example MLE-02. 

Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind as a potential  predictive model. When the model is based on a linear equation, such as
$$Y = b_0 + b_1X_1 + b_2X_2 +_ \cdots + b_kX_k,$$
we have **linear regression**. $b_0$ is the **intercept** and $b_1, b_2, \dots, b_k$ are the **regression coefficients**.

Though the predictions of a linear regression model can usually be improved with more advanced techniques, most analysts start there, because it helps them to understand the data. Also, a linear equation can be understood and discussed by those who know a bit of statistics, which may be interesting in some applications, in which the **interpretability** of the model is relevant. Alternative approaches will be discussed later in this course.

## Prediction error

In general, regression models are evaluated through their **prediction errors**. The basic schema is
$$\textrm{Prediction\ error} = \textrm{Actual\ value} + \textrm{Predicted\ value}.$$

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the **mean squared error** (MSE) is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in regression models obtained by other methods. 

## R-squared

The **coefficient of determination** is a popular metric for the evaluation of regression models. It is given by the formula
$$R^2 = 1 - \displaystyle \frac{\textrm{MSE}}{{\textrm{MSY}}}\thinspace,$$ 
in which MSY is the average squared centered target value (subtracting the mean). It can be proved that $0 \le R^2 \le 1$. The maximum is $R^2 = 1$, which is equivalent to a null MSE, which happens when all the errors are zero, and all the predicted values are exactly equal to the corresponding actual values.

The statistics textbook explains the coefficient of determination as the **percentage of variance** explained by the regression equation. Also, it coincides with the square of the correlation between the actual and the predicted values, called the **multiple correlation**. This explains the notation, and the name **R-squared statistic**, given to the coefficient of determination. Since MSY is fixed, minimizing MSE is equivalent to maximixing the correlation between actual and predicted values. 

Nevertheless, this is no longer true in regression models which are nonlinear or that, though being linear, are not obtained by the least squares method. So, in spite of the R-squared statistics being presented as the standard metric for regression models in many sources, it is recommended to use a simple and direct way for comparing actual and predicted target values. Some examples could be a scatter plot and correlation analysis of the actual and predicted target values, or a metric based on the prediction error such as the **mean absolute error** or the **mean absolute percentage error**, whose interpretation is completely straightforward.

A final word of caution. Correlation and R-squared are sensitive to extreme values, not unfrequent in real-world data. So, when the target has a skewed distribution, with a long right tail, a strong correlation may suggest that the model is better than it really is. Example MLE-02 illustrates this point.
