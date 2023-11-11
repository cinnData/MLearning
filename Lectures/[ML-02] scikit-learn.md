# [ML-02] scikit-learn

## What is scikit-learn?

The package **scikit-learn** (sklearn in the code) is a machine learning toolkit, built on top of NumPy, SciPy and Matplotlib. To get an idea of the hierarchy and the contents of the various scikit-learn subpackages, the best source is the **scikit-learn API Reference** (`scikit-learn.org/stable/modules/classes.html`). Some of these subpackages will appear in this course: `linear_model`, `tree`, `metrics`, `ensemble`, etc.

In Python, a **class** is like an object constructor, or a "blueprint" for creating objects. The subpackages that we use for supervised learning contain a collection of **estimator classes**, which allow us to create and apply predictive models. In this course, we will find a number of these classes: `LinearRegression()`, `LogisticRegression()`, `DecisionTreeClassifier()`, etc.

The scikit-learn API provides code guidelines which are quite consistent across the different estimator classes. We see an example in this lecture. The first time you will find it a bit awkward, but you will get used after some practice.

Working with scikit-learn, you may receive a **warning** from time to time. Note that a warning is not the same as an **error message**. An error message stops the execution of your command, while a warning does not. Most of the warnings will tell you nothing of interest, but a few ones contain relevant information, so it is recommended to take a look at them with the corner of your eye. This is a price to pay for working in such a dynamic field.

## Supervised learning in scikit-learn

To train a supervised learning method in scikit-learn, you have to specify a (1D) **target vector** `y` and a (2D) **feature matrix** `X`. In regression, both `X` and `y` have to be numeric or Boolean (type `str` is converted on the fly to `float`, when possible), but, in classification, `y` can be a string vector. Both NumPy arrays and Pandas data containers are accepted, but the scikit-learn methods always return NumPy arrays.

The first step is to import the class you wish to use from the corresponding subpackage. For instance, to train a linear regression model, you will start by:

```
from sklearn.linear_model import LinearRegression
```

Your estimator will be an **instance** of this class, that is, an object which applies the methodology chosen. For linear regression, we use:

```
model = LinearRegression()
```

Here, `model` is a name chosen by the user. Note that, leaving the parenthesis empty, we accept the **default arguments**. This makes sense for linear regression, but it will be wrong for decision trees, where we typically control the growth, to prevent overfitting.

## The three basic methods

Irrespective of the type of predictive model, three basic methods, namely `.fit()`, `.predict()` and `.score()`, are available. The method `fit` carries out the training, that is, it finds the model that works best for the data, within the class selected. Training is always based on minimizing a **loss function**. In the default option of `LinearRegression()`, and in many other regression classes in scikit-learn, the loss function is the **mean squared error**. More detail will be provided in the next lecture.

The analysis would start as:

```
model.fit(X, y)
```

Once the estimator has fitted the data, the **predicted values** are extracted with the method `predict()`:

```
y_pred = model.predict(X)
```

Finally, the method `.score()` provides an assessment of the quality of the predictions, that is, of the match between `y` and `y_pred`:

```
model.score(X, y)
```

In both regression and classification, `.score()` returns a number whose maximum value is 1, which is read as *the higher the better*. Nevertheless, the mathematics are completely different. For a regression model, it is the **R-squared statistic**. For a classification model, it is the **accuracy**. Details will be given in the following lectures.

## Dummy variables

Statisticians refer to the binary variables that take 1/0 values as **dummy variables**, or dummies. They use dummies to enter **categorical variables**, whose values are labels for groups or categories, in regression equations. Note that a categorical feature can have a numeric data type (*e.g*. the zipcode) or string type (*e.g*. the gender, coded as F/M). Since dummies are frequently used in data analysis, statistical software applications provide simple ways for extracting them. 

In machine learning, **one-hot encoding** is a popular name for the process of extracting dummies from a set of a categorical features. scikit-learn has a method for encoding categorical features in a massive way, for several features in one shot. 

In applications like Excel, where we explicitly create new columns for the dummies to carry out a regression analysis, the rule is that the number of dummies is equal to the number of groups minus one. We set one of the groups as the **baseline group** and create one dummy for every other group. More specifically, suppose that $G_0$ is the baseline group and $G_1$, $\dots$ , etc, are the other groups. First, we define a dummy $D_1$ for $G_1$ as $D_1 = 1$ on the samples from group $G_1$, and $D_1 = 0$ on the samples from the other groups. Then, we repeat this for $G_2$, and the rest of the groups except the baseline, which leads to a set of dummies $D_1$, $D_2$, etc, which can be included in the equation.

In Python, we do not have to care about all this. We just pack in a matrix (2D array or Pandas data frame) the categorical features that you wish to encode as dummies and apply to that matrix the appropriate method, explained below, which returns a new matrix whose columns are the dummies.

## One-hot encoding in scikit-learn

In scikit-learn, a one-hot encoding **transformer** can be extracted from the class `OneHotEncoder()` of the subpackage `preprocessing`. Suppose that you pack the features that we wish to encode as a matrix `X2`, and the rest of the features as a separate matrix `X1`. Thus, `X2` is transformed as follows. 

First, we instantiate a `OneHotEncoder` transformer as usual in scikit-learn:

```
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
```

Next, we fit the transformer `enc` to the feature matrix `X2`:

```
enc.fit(X2)
```

You get the encoded matrix as:

```
X2 = enc.transform(X2).toarray()
```

Finally, we put the two feature matrices together with the NumPy function `concatenate()`:

```
X = np.concatenate([X1, X2], axis=1)
```

`axis=1` indicates that the two matrices are **concatenated** horizontally. For this to be possible, they must have the same number of rows. The default of `concatenate()` is `axis=0`, that is, vertical concatenation.

*Note*. Instead of `concatenate()`, you can use here `hstack()` for the same purpose 

## One-hot encoding in Pandas

One-hot encoding can also be performed with the Pandas function `get_dummies()`. Suppose that the feature matrix is split in two two data frames X1 and X2, as recommendedin the preceding section. The code is then simpler than in scikit-learn:

```
X2 = pd.get_dummies(X2)
X = pd.concat([X1, X2], axis=1)
```

Note that both `get_dummies()` and `concat()` take only Pandas objects, returning a Boolean Pandas data frame. An advantage of using Pandas is that each column of the matrix of dummies comes with an intelligible name.

*Note*. Instead of `concat([X1, X2], axis=1)`, you can use `X1.merge(X2)` for the same purpose.

## Saving a scikit-learn model

How can you save your model, to use it in another session, without having to train it again? This question is capital in business applications, where you use the model to predict a target value for new samples for which the target has not yet been observed. Of course, if your model is a simple linear regression equation, you can extract the coefficients of the regression equation, write the equation and apply it to the incoming samples.

But, even if this seems feasible for a simple equation, it would not be so for the more complex models, which may look like black boxes to you. There are many ways to save and reload an object in Python, but the recommended method for scikit-learn models is based on the functions `dump()` and `load()` of the package `joblib`, included in the Anaconda distribution. This package uses a special file format, the **PKL file format** (extension `.pkl`).

With `joblib`, saving your model to a PKL file is straightforward. For our model above, this would be:

```
import joblib
joblib.dump(model, 'model.pkl')
```

Do not forget to add the path for the PKL file. You can recover the model, anytime, even if you no longer have the training data, as:

```
newmodel = joblib.load('model.pkl')
```
