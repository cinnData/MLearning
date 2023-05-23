# [ML-02] scikit-learn

## What is scikit-learn?

The package **scikit-learn** (sklearn in the code) is a machine learning toolkit, built on top of NumPy, SciPy and Matplotlib. To get an idea of the hierarchy and the contents of the various scikit-learn subpackages, the best source is the **scikit-learn API Reference** (`scikit-learn.org/stable/modules/classes.html`). Some of these subpackages will appear in this course: `linear_model`, `tree`, `metrics`, `ensemble`, etc.

In Python, a **class** is like an object constructor, or a "blueprint" for creating objects. The subpackages that we use for supervised learning contain a collection of **estimator classes**, or ways to create and apply predictive models. In this course you will see a number of these classes: `LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, etc.

The scikit-learn API provides rules for writing your code which are quite consistent across the different estimator classes. We see an example in this lecture. The first time you will find it a bit awkward, but you will get used after some practice.

Working with scikit-learn, you will get **warnings** from time to time. Note that a warning is not the same as an **error message**. An error message stops the execution of your command, while a warning does not. Most of the warnings will tell you nothing of interest, but a few ones contain relevant information, so it is recommended to take a look at them with the corner of your eye. This is a price to pay for working in such a dynamic field. Beyond scikit-learn (*e.g*. with TensorFlow)  

## Supervised learning in scikit-learn

To train a supervised learning method in scikit-learn, you have to specify a (1D) **target vector** `y` and a (2D) **feature matrix** `X`. In regression, both `X` and `y` have to be numeric or Boolean (type `str` is converted to `float` on the fly), but, in classification, `y` can be a string vector. Both NumPy arrays and Pandas data containers (series and data frames) are accepted, but the scikit-learn methods always return NumPy arrays.

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

Irrespective of the type of predictive model, three basic methods, namely `fit`, `predict` and `score`, will be available. The method `fit` carries out the training, that is, it finds the model that works best for the data, within the class selected. It is always based on minimizing a **loss function**. In the default option of `LinearRegression`, and in many other regression classes in scikit-learn, the loss function is the **mean squared error**. More detail will be provided in the next lecture.

The syntax would be:

```
model.fit(X, y)
```

Once the estimator has fitted the data, the **predicted values** are extracted with the method `predict`:

```
y_pred = model.predict(X)
```

Finally, the method `score` provides an assessment of the quality of the predictions, that is, of the match between `y` and `y_pred`:

```
model.score(X, y)
```

In both regression and classification, score returns a number whose maximum value is 1, which is read as *the higher the better*. Nevertheless, the mathematics are completely different. For a regression model, it is the **R-squared statistic**. For a classification model, it is the **accuracy**. Details will be given in the following lectures.

## Dummy variables

Statisticians refer to the binary variables that take 1/0 values as **dummy variables**, or dummies. They use dummies to enter **categorical variables**, whose values are labels for groups or categories, in regression equations. Note that a categorical feature can have a numeric data type (*e.g*. the zipcode) or string type (*e.g*. the gender, coded as F/M). Since dummies are frequently used in data analysis, statistical software applications provide simple ways for extracting them. 

In machine learning, **one-hot encoding** is a popular name for the process of extracting dummies from a set of a categorical features. scikit-learn has a method for encoding categorical features in a massive way, for several features in one shot. 

In applications like Excel, where you explicitly create new columns for the dummies to carry out a regression analysis, the rule is that the number of dummies is equal to the number of groups minus one. You set one of the groups as the **baseline group** and create one dummy for every other group. More specifically, suppose that $G_0$ is the baseline group and $G_1$, $\dots$ , etc, are the other groups. First, you define a dummy $D_1$ for $G_1$ as $D_1 = 1$ on the samples from group $G_1$, and $D_1 = 0$ on the samples from the other groups. Then, you repeat this for $G_2$, and the rest of the groups except the baseline, which leads to a set of dummies $D_1$, $D_2$, etc, which you can include in your equation.

Using Python, you do not have to care about all this. You just pack in a 2D array the categorical features that you wish to encode as dummies and apply to that array the appropriate method, explained below, which returns a new array whose columns are the dummies.

## One-hot encoding in scikit-learn

In scikit-learn, a one-hot encoding **transformer** can be extracted from the class `OneHotEncoder` from the subpackage `preprocessing`. Suppose that you pack the features that we wish to encode as a matrix `X2`, and the rest of the features as a separate matrix `X1`. Thus, `X2` is transformed as follows. 

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

*Note*. Instead of `concatenate()`, you can use `np.hstack((X1, X2)` for the same purpose 

## One-hot encoding in Pandas

One-hot encoding can also be performed with the Pandas function `get_dummies()`. Suppose that the feature matrix is split in two two data frames X1 and X2, as in the preceding section. The code is then simpler than in scikit-learn:

```
X2 = pd.get_dummies(X2)
X = pd.concat([X1, X2], axis=1)
```

Note that both `get_dummies()` and `concat()` only take Pandas objects, returning a Pandas object. An advantage of using Pandas is that each column of the matrix of dummies comes with an intelligible name.

*Note*. Instead of `concat()`, you can use `X1.merge(X2)` for the same purpose.

## Saving a scikit-learn model

How can you save your model, to use it in another session, without having to train it again? This question is capital for the applicability of the model in business, where you use it to predict a target value for new samples for which the target has not yet been observed. Of course, if your model is a simple linear regression equation, you can extract the coefficients of the regression equation, write the equation and apply it to the incoming samples.

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

## Example - Modeling the strength of concrete

In the field of engineering, it is crucial to have accurate estimates of the performance of building materials. These estimates are required in order to develop safety guidelines governing the materials used in the construction of buildings, bridges, and roadways.

Estimating the strength of concrete is a challenge of particular interest. Although it is used in nearly every construction project, concrete performance varies greatly due to a wide variety of ingredients that interact in complex ways. As a result, it is difficult to accurately predict the strength of the final product. A model that could reliably predict concrete strength given a listing of the composition of the input materials could result in safer construction practices.

The concrete compressive strength seems to be a highly nonlinear function of age and ingredients. I-Cheng Yeh found success using neural networks to model concrete strength data. The file `concrete.csv` contains 1,030 examples of concrete with eight features describing the components used in the mixture. These features are thought to be related to the final compressive strength and they include the amount (in kilograms per cubic meter) of cement, slag, ash, water, superplasticizer, coarse aggregate, and fine aggregate used in the product in addition to the aging time (measured in days). 

The variables included in the data set are:

* `cement`, cement concentration in the mixture (kg/m3).

* `slag`, blast furnace slag concentration in the mixture (kg/m3).

* `ash`, fly ash concentration in the mixture (kg/m3).

* `water`, water concentration in the mixture (kg/m3).

* `superplastic`, superplasticizer concentration in the mixture (kg/m3).

* `coarseagg`, coarse aggregate concentration in the mixture (kg/m3).

* `fineagg`, fine aggregate concentration in the mixture (kg/m3).

* `age`, age in days.

* `strength`, concrete compressive strength (MPa).

*Source*. I-Cheng Yeh (1998), Modeling of strength of high performance concrete using artificial neural networks, *Cement and Concrete Research* **28**(12), 1797-1808.

We use the Pandas funcion `read_csv()` to import the data. First, we import the package:

```
In [1]: import pandas as pd
```

We use a remote path, to a GitHub repository. None of the columns is used as the index.

```
In [2]: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'concrete.csv')
````

```
In [3]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1030 entries, 0 to 1029
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   cement        1030 non-null   float64
 1   slag          1030 non-null   float64
 2   ash           1030 non-null   float64
 3   water         1030 non-null   float64
 4   superplastic  1030 non-null   float64
 5   coarseagg     1030 non-null   float64
 6   fineagg       1030 non-null   float64
 7   age           1030 non-null   int64  
 8   strength      1030 non-null   float64
dtypes: float64(8), int64(1)
memory usage: 72.5 KB
```

```
In [4]: df.head()
Out[4]: 
   cement   slag  ash  water  superplastic  coarseagg  fineagg  age  strength
0   540.0    0.0  0.0  162.0           2.5     1040.0    676.0   28     79.99
1   540.0    0.0  0.0  162.0           2.5     1055.0    676.0   28     61.89
2   332.5  142.5  0.0  228.0           0.0      932.0    594.0  270     40.27
3   332.5  142.5  0.0  228.0           0.0      932.0    594.0  365     41.05
4   198.6  132.4  0.0  192.0           0.0      978.4    825.5  360     44.30
```

```
In [5]: y = df.iloc[:, -1]
   ...: X = df.iloc[:, :-1]
```

Alternatively, you can use the names of the columns, setting `y = df['strength']` and `X = df.drop(columns='strength')`.

```
In [6]: from sklearn.linear_model import LinearRegression
   ...: model = LinearRegression()
```

```
In [7]: model.fit(X, y)
Out[7]: LinearRegression()
```

```
In [8]: y_pred = model.predict(X)
```

```
In [9]: round(model.score(X, y), 3)
Out[9]: 0.616
```

```
In [10]: from matplotlib import pyplot as plt
```

```
In [11]: plt.figure(figsize=(6,6))
    ...: plt.scatter(y_pred, y, color='black', s=2)
    ...: plt.xlabel('Predicted strength')
    ...: plt.ylabel('Actual strength');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_2.1.png)

