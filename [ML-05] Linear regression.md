# [ML-05] Linear regression

## scikit-learn

The package **scikit-learn** (sklearn in the code) is a machine learning toolkit, built on top of NumPy, SciPy and Matplotlib. To get an idea of the hierarchy and the contents of the various scikit-learn subpackages, the best source is the **scikit-learn API Reference** (`scikit-learn.org/stable/modules/classes.html`). Some of these subpackages will appear in this course: `linear_model`, `tree`, `metrics`, `ensemble`, etc.

In Python, a **class** is like an object constructor, or a "blueprint" for creating objects. The subpackages that we use for supervised learning contain_ a collection of **estimator classes**, or ways to create and apply predictive models. In this course you will see a number of these classes: `LinearRegression`, `LogisticRegression`, `cDecisionTreeClassifier`, etc.

The scikit-learn API provides rules for writing your code which are quite consistent across the different estimator classes. We see how this works for linear regression in this lecture. The first time you will find it a bit awkward, but you will get used after some examples.

Working with scikit-learn, you will get **warnings** from time to time. Note that a warning is not the same as an **error message**. An error message stops the execution of your command, while a warning does not. Most of the warnings will tell you nothing of interest, but a few ones may be discussed in the examples of this course.

## Linear regression

In machine learning, the term **regression** applies to the prediction of a numeric target. Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind as a potential  predictive model.

When the model is based on a linear equation, as in

$$Y = b_0 + b_1X_1 + b_2X_2 +_ \cdots + b_kX_k,$$

we have **linear regression**, which is the subject of this lecture. Though the predictions of a linear regression model can usually be improved with more advanced techniques, most analysts start there, because it helps them to understand the data. An alternative approach, based on **decision trees**, will be discussed later in this course.

## Evaluation of a linear regression model

In general, regression models are evaluated through their **prediction errors**. The basic schema is

$$\textrm{Prediction\ error} = \textrm{Actual\ value} + \textrm{Predicted\ value}.$$

The **coefficient of determination** is a popular metric for the evaluation of regression models. It is given by the formula

$$R^2 = 1 - \displaystyle \frac{\textrm{SSE}}{{\textrm{SSY}}}\,,$$ 

in which SSE is the sum of the squared errors and SSY is the sum of the squared centered target values (subtracting the mean).

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the **mean squared error** (MSE) is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in regression models obtained by other methods. In that case, the coefficient of determination can be interpreted as the **percentage of variance** explained by the equation. Moreover, it coincides with the square of the correlation between the actual and the predicted values, called the **multiple correlation** in statistics textbooks. This explains the notation, and the name **R-squared statistic**, given to the coefficient of determination.

Nevertheless, this is no longer true in regression models which are nonlinear or that, though being linear, are not obtained by the least squares method. So, in spite of the R-squared statistics coming as the standard metric for regression models, I would recommend you to use the correlation of actual and predicted values or a metric based on the prediction error such as the **mean absolute error** or the **mean absolute percentage error**, whose interpretation is completely straightforward.

## Evaluation of a linear regression model

To train a supervised learning method in scikit-learn, you have to specify a (1D) **target vector** `y` and a (2D) **feature matrix** `X`. In regression, both `X` and `y` have to be numeric or Boolean (type `str` is converted to `float` on the fly), but, in classification, `y` can be a string vector. 

The first step is to import the class you wish to use from the corresponding subpackage. For instance, to train a linear regression model, you will start by:

```
from sklearn.linear_model import LinearRegression
```

Your estimator will be an **instance** of this class, that is, an object which applies the methodology chosen. For linear regression, we use:

```
reg = LinearRegression()
```

Note that `reg` is a name chosen as a reminder of what this object is. You can call it what you want. Also, note the parenthesis in this definition. Leaving the parenthesis empty, you accept the **default arguments**. This makes sense for linear regression, where the default is the old least squares method, but it will be wrong for decision trees, where we typically control the growth to prevent overfitting.

## The three basic methods

Irrespective of the type of predictive model, three basic methods, namely fit, predict and score, are available. The method `fit` carries out the training, that is, it finds the model that works best for the data, within the class selected. Here, the syntax would be:

```
reg.fit(X, y)
```

Depending on the type of model, `fit` will do different jobs. For `reg`, the prediction is obtained by means of a linear equation, so `fit` finds the optimal coefficients for the equation. This means that the model extracted is the one for which the match between the actual target values (the vector `y`) and the predicted target values (the vector `y_pred` below) is the best possible. If we use the linear regression default, this is operationalized by the least squares method.

The method `fit` is always based on minimizing a **loss function**. In the default option of `LinearRegression`, and in many other regression classes in scikit-learn, the loss function is the mean squared error.

Once the estimator has fitted the data, the predicted values are extracted with the method `predict`:

```
ypred = reg.predict(X)
```

Finally, the method `score` provides an assessment of the quality of the predictions, that is, of the match between `y` and `y_pred`:

```
reg.score(X, y)
```

In both regression and classification, score returns a number whose maximum value is 1, which is read as *the higher the better*. Nevertheless, the mathematics are completely different. For a regression model, it is the R-squared statistic. For a classification model, it is the accuracy, to be explained in the next lecture.

## Dummy variables

Statisticians refer to the binary variables that take 1/0 values as **dummy variables**, or dummies. They use dummies to enter **categorical variables**, whose values are labels for groups or categories, in regression equations. Note that a categorical feature can have a numeric data type (*e.g*. the zipcode) or string type (*e.g*. the gender, coded as F/M). Since dummies are frequently used in data analysis, statistical software applications provide simple ways for extracting them. 

In machine learning, **one-hot encoding** is a popular name for the process of extracting dummies from a set of a categorical features. scikit-learn has a method for encoding categorical features in a massive way, for several features in one shot. 

In applications like Excel, where you explicitly create new columns for the dummies to carry out a regression analysis, the rule is that the number of dummies is equal to the number of groups minus one. You set one of the groups as the **baseline group** and create one dummy for every other group. More specifically, suppose that $G_0$ is the baseline group and $G_1$, $\dots$ , etc, are the other groups. First, you define a dummy $D_1$ for $G_1$ as $D_1 = 1$ on the samples from group $G_1$, and $D_1 = 0$ on the samples from the other groups. Then, you repeat this for $G_2$, and the rest of the groups except the baseline, which leads to a set of dummies $D_1$, $D_2$, etc, which you can include in your equation.

Using Python, you do not have to care about all this. You just pack in a 2D array the categorical features that you wish to encode as dummies and apply to that array the appropriate method, explained below, which returns a new array whose columns are the dummies.

## One-hot encoding in scikit-learn

In scikit-learn, a one-hot encoding **transformer** can be extracted from the class `OneHotEncoder` from the subpackage `preprocessing`. Suppose that you pack the features that we wish to encode as a matrix `X2`, and the rest of the features as a separate matrix `X1`. Thus, `X2` is transformed as follows. 

First, we instantiate a OneHotEncoder transformer as usual in scikit-learn:

```
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
```

Next, you fit the transformer `enc` to the feature matrix `X2`:

```
enc.fit(X2)
```

You get the encoded matrix as:

```
X2 = enc.transform(X2).toarray()
```

Finally, you put the two feature matrices together with the NumPy function `concatenate`:

```
X = np.concatenate([X1, X2], axis=1)
```

`axis=1` indicates that the two matrices are **concatenated** horizontally. For this to be possible, they must have the same number of rows. The default of concatenate is `axis=0`, that is, vertical concatenation.

## Saving a scikit-learn model

How can you save your model, to use it in another session, without having to train it again? This question is capital for the applicability of the model in business, where you use it to predict a target value for new samples for which the target has not yet been observed. Of course, if your model is a simple linear regression equation, you can extract the coefficients of the regression equation, write the equation and apply it to the incoming samples.

But, even if this seems feasible for a simple regression equation, it would not be so for the more complex models, which may look like black boxes to you. There are many ways to save and reload an object in Python, but the recommended method for scikit-learn models is based on the functions `dump` and `load` of the package `joblib`, included in the Anaconda distribution. This package uses a special file format, the **PKL file format** (extension `.pkl`).

With `joblib`, saving your model to a PKL file is straightforward. For our linear regression model `reg`, this would be:

```
import joblib
joblib.dump(reg, 'reg.pkl')
```

Do not forget to add the path for the PKL file. You can recover the model, anytime, even if you no longer have the training data, as:

```
newreg = joblib.load('reg.pkl')
``
