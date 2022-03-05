# 5. Linear regression

### scikit-learn

The package scikit-learn (`sklearn` in the code) is a machine learning toolkit, built on top of NumPy, SciPy and Matplotlib. To get an idea of the hierarchy and the contents of the various scikit-learn subpackages, the best source is the **scikit-learn API Reference** (`scikit-learn.org/stable/modules/classes.html`). Some of these subpackages will appear in this course: `linear_model`, `tree`, `metrics`, `cluster`, etc.

In Python, a **class** is like an object constructor, or a "blueprint" for creating objects. The subpackages that we use for supervised learning contain a collection of **estimator classes**, or ways to create and apply predictive models. In this course you will see a number of these classes: `LinearRegression`, `LogisticRegression`, `DecisionTreeRegressor`, etc.

The scikit-learn API provides rules for writing your code which are quite consistent across the different estimator classes. We see how this works for linear regression in this chapter. The first time you will find it a bit awkward, but you will get used after some examples.

Working with scikit-learn, you will get **warnings** from time to time. Note that a warning is not the same as an **error message**. An error message stops the execution of your command, while a warning does not. Some of the warnings tell you nothing of interest, but some of them will be discussed in this course.

### Linear regression

In machine learning, the term **regression** applies to the prediction of a numeric target. Regression models are not necessarily related to a mathematical equation, as in statistical analysis, although an equation is the first idea that comes to our mind when we think on a "predictive model".

When the model is based on a linear equation, as in

<img src="https://render.githubusercontent.com/render/math?math=\large Y = b_0 %2B b_1 X_1 %2B b_2 X_2 %2B \cdots %2B b_k X_k,">

we have **linear regression**, which is the subject of this chapter. Though the predictions of a linear regression model can usually be improved by more sophisticated techniques, most analysts start there, because it helps them to understand the data. An alternative approach, based on **decision trees**, will be discussed later in this course.

### Evaluation of a linear regression model

In general, regression models are evaluated through their prediction errors. The basic schema is

<img src="https://render.githubusercontent.com/render/math?math=\large \textrm{Prediction\ error} = \textrm{Actual\ value} - \textrm{Predicted\ value}.">

In a linear regression context, prediction errors are called **residuals**. In the standard approach to linear regression, the regression coefficients are calculated so that the sum of the squared residuals is minimum. This is called the **least squares method**. The errors of a linear equation obtained by means of the least squares method have an important property, that their sum is zero, which is no longer true in other regression models.

Statisticians look at the **residual sum of squares** for evidence of good fit between the regression model and the data. The *R* **squared statistic** is a standardized measure which operationalizes this. More specifically, they take advantage of the variance decomposition formula

<img src="https://render.githubusercontent.com/render/math?math=\large R^2 = \displaystyle \frac{\textrm{var(Predicted\ values)}} {\textrm{var(Actual\ values)}}\,.">

It turns out that the square root of this proportion coincides with the correlation between the actual and the predicted values, called the **multiple correlation** in statistics textbooks. Although this is no longer true for other regression methods, this correlation is still the standard approach to the evaluation of a regression model.

### Linear regression in scikit-learn

To train a supervised learning method in scikit-learn, you have to specify a (1d) **target vector** `y` and a (2d) **feature matrix** `X`. In regression, both `X` and `y` have to be numeric, but, in classification, `y` can be a string vector. 

The first step  you have to import the class you wish to use from the corresponding subpackage. For instance, to train a linear regression model, you will start by:

`from sklearn.linear_model import LinearRegression` 

Your estimator will be an **instance** of this class, that is, an object which applies the methodology chosen. For linear regression, we use:

`linreg = LinearRegression()`

Note that `linreg` is a name which I use here to remind us what this object is. You can call it what you want. Also, note the parenthesis in this definition. Leaving the parenthesis empty, you accept the **default arguments**. This makes sense for linear regression, where the default is the old least squares method, but it may not for other models like a decision tree, in which we typically control the growth to prevent overfitting. 

### The three basic methods

Irrespective of the type of predictive model, three basic methods, namely `fit`, `predict` and `score`, are available. The method `fit` does the training, that is, it finds the parameters of the predictive model that work best for the data. Here, the syntax would be:

`linreg.fit(X, y)`

Depending on the type of model, `fit` will do different jobs. For `linreg`, the prediction is obtained by means of a linear equation, so `fit` finds the optimal coefficients for the equation. This means that the model extracted is the one for which the match between the **actual target values** (the vector `y`) and the **predicted target values** (the vector `y_pred` below) is the best possible. If we use the linear regression default, this is operationalized by the least squares method.

Once the estimator has fitted the data, the predicted values are obtained with the method `predict`:

`ypred = linreg.predict(X)`

Finally, the method `score` provides an assessment of the quality of the predictions, that is, of the match between `y` and `ypred`: 

`linreg.score(X, y)`

In both regression and classification, `score` returns a number in the 0-1 range, which is read as *the higher the better*. Nevertheless, the matematics are completely different. For a regression model, it is a *R*-squared statistic, that is the squared correlation of `y` and `ypred`.

### Saving a scikit-learn model

How can you save your model, to use it in another session, without having to train it again? This question is capital for the applicability of the model in business, where you use it to predict a target value for new samples for which the target has not yet been observed. Of course, if your model is a simple linear regression equation, you can extract the coefficients of the regression equation, write the equation and apply it to the incoming samples. 

But, even if this seems feasible for a simple regression equation, it would not be so for the more complex models, which may look like black boxes to you. There are many ways to save and reload an object in Python, but the recommended method for scikit-learn models is based on the functions `dump` and `load` of the package `joblib`, which is included in the Anaconda distribution. This package uses a special file format, the **PKL file format** (extension `.pkl`).

With `joblib`, saving your model to a PKL file is straightforward. For our linear regression model `linreg`, this would be:

`import joblib`

`joblib.dump(linreg, 'linreg.pkl')`
   
Do not forget to add the path for the PKL file. You can recover the model, anytime, even if you no longer have the training data, with:

`newlinreg = joblib.load('linreg.pkl')`
