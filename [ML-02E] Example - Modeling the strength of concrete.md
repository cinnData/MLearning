# [ML-02E] Example - Modeling the strength of concrete

## Introduction

The purpose of this example is to illustrate the basic syntax of scikit-learn using data from the field of engineering. In that filed, it is crucial to have accurate estimates of the performance of building materials. These estimates are required in order to develop safety guidelines governing the materials used in the construction of buildings, bridges and roadways.

Estimating the strength of concrete is a challenge of particular interest. Although it is used in nearly every construction project, concrete performance varies greatly due to a wide variety of ingredients that interact in complex ways. As a result, it is difficult to accurately predict the strength of the final product. A model that could reliably predict concrete strength given a listing of the composition of the input materials could result in safer construction practices.

The concrete compressive strength seems to be a highly nonlinear function of age and ingredients. I-Cheng Yeh found success using neural networks to model concrete strength data. The file `concrete.csv` contains 1,030 examples of concrete with eight features describing the components used in the mixture. These features are thought to be related to the final compressive strength and they include the amount (in kilograms per cubic meter) of cement, slag, ash, water, superplasticizer, coarse aggregate, and fine aggregate used in the product in addition to the aging time (measured in days). 

## The data set

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

Source: I-Cheng Yeh (1998), Modeling of strength of high performance concrete using artificial neural networks, *Cement and Concrete Research* **28**(12), 1797-1808.

## Questions

Q1. Import this data set to a Pandas data frame and check that the content matches the descriptio given above. 

Q2. Prepare the data for supervised learning by creating a **target vector** and a **feature matrix**.

Q3. Develop a **linear regression model** for predicting the strength in terms of the components of the mixture.

Q4. Use the model to obtain predicted strength values.

Q5. Evaluate the model obtained.

Q6. Save the model for future use.

## Q1. Import the data

Although scikit-learn is described in the technical documentation as managing the data in NumPy array format, you can equally input data in Pandas format. Using Pandas format makes processing slower, but importing the data and adapting them for the learning process will be easier for you. Nevertheless, remember that, in machine learning, preprocessing is a previous step, so the learning process is carried out with NumPy arrays. Second, that even if scikit-learn estimators can take Pandas data containers, it aleways return arrays. 

We use here the Pandas funcion `read_csv()` to import the data. First, we import the package:

```
In [1]: import pandas as pd
```

The source file is in a GitHub repository, so we use a remote path to get access. 

```
In [2]: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'concrete.csv')
````

`df` is a Pandas data frame. Since we have used the default of `read_csv()`, none of the columns is used as the index. To explore the data set, we use the standard Pandas methods. First,  the method `.info()` prints a report of the data frame content. We can see that all the columns are numeric, and that there are no missing values.

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

The method `.head()` extracts the first five rows. Everything looks right, so far.

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

Finally, the method `.describe()` produces a statistical summary. The distribution of `strength` looks quite symmetric, which makes linear regression results easier to manage. The same is not true for all the independent variables, specially for `ash`.

```
In [5]: df.describe()
Out[5]: 
            cement         slag          ash        water  superplastic  \
count  1030.000000  1030.000000  1030.000000  1030.000000   1030.000000   
mean    281.167864    73.895825    54.188350   181.567282      6.204660   
std     104.506364    86.279342    63.997004    21.354219      5.973841   
min     102.000000     0.000000     0.000000   121.800000      0.000000   
25%     192.375000     0.000000     0.000000   164.900000      0.000000   
50%     272.900000    22.000000     0.000000   185.000000      6.400000   
75%     350.000000   142.950000   118.300000   192.000000     10.200000   
max     540.000000   359.400000   200.100000   247.000000     32.200000   

         coarseagg      fineagg          age     strength  
count  1030.000000  1030.000000  1030.000000  1030.000000  
mean    972.918932   773.580485    45.662136    35.817961  
std      77.753954    80.175980    63.169912    16.705742  
min     801.000000   594.000000     1.000000     2.330000  
25%     932.000000   730.950000     7.000000    23.710000  
50%     968.000000   779.500000    28.000000    34.445000  
75%    1029.400000   824.000000    56.000000    46.135000  
max    1145.000000   992.600000   365.000000    82.600000  
```

## Q2. Target vector and feature matrix

Looking at this data set with a supervised learning perspective, we specify the concrete strength as the **target vector**, that is, what we wish to predict. To be consistent, we will always denote this vector (it could either a NumPy 1D array or a Pandas series) as `y`. In this case, the target vector is the last column, which we can extract as:

```
In [6]: y = df.iloc[:, -1]
```

Alternatively, you can use the column name, setting `y = df['strength']`. The feature matrix packs the rest of the columns:

```
In [7]: X = df.iloc[:, :-1]
```

The same can be obtained as `X = df.drop(columns='strength')`.

## Q3. Linear regression model

To obtain a **linear regression model** for this example, we use the scikit-learn class `LinearRegression`, from the subpackage `linear_model`. First, we import this class:

```
In [8]: from sklearn.linear_model import LinearRegression
```

We create an instance of this estimator with:

```
In [9]: reg = LinearRegression()
```

The method `.fit()` returns the "model", that is an object that, given the components of the mixture, returns the predicted strength. In this case, the model consists in a linear equation, so what `.fit()` does is to calculate the coefficients for the equation. These coefficients are optimal (minimum MSE) for the data provided (the pair `X`, `y`).

```
In [10]: reg.fit(X, y)
Out[10]: LinearRegression()
```

## Q4. Predicted strength values

Given the feature values, the method `.predict()` returns predicted target values. The argument must a numeric 2D array or a data frame with 8 columns. We can use it on the feature matrix used in the learning process: 

```
In [11]: y_pred = reg.predict(X)
```

The vector `y_pred` of predicted values can be compared to the vector `y` of actual values, to evaluate the predictive performance of the model. But it can also be applied to new data, which is what we want the model for in real applications. Let us do it in a fictional new sample, which we create by setting the feature values obtained by rounding the mean values calculated in `Out [6]`. 

```
In [14]: X_new = df.describe().iloc[1:2, :-1].round()
    ...: X_new
Out[14]: 
      cement  slag   ash  water  superplastic  coarseagg  fineagg   age
mean   281.0  74.0  54.0  182.0           6.0      973.0    774.0  46.0
```

Remember that the argument of `.predict()` must be two-dimensional. That is why we use `iloc[1:2, :]` to select the rows in the definition of `X_new`. With `iloc[1:2, :]`, you get a series instead of a data frame. Now:

```
In [15]: reg.predict(X_new)
Out[15]: array([35.71595681])
```

The predicted strength, for this mixture, would be 35.72 MPa. Note that that the output comes as a 1D array. Adding extra rows to `X_new` would give extra length to this array. Also, note that the predicted strength is very close to mean strength in the training data. This is a property of linear regression models.

## Q5. Evaluate the model

The method `.score()` provides a quick and dirty evaluation of the model: 

```
In [16]: round(reg.score(X, y), 3)
Out[16]: 0.616
```

In the case of a regression model, the value returned by this method is **R-squared statistic**, which you can see as a squared correlation (this is exactly so only in linear regression), the correlation between the actual strength (`y`) and the predicted (`y_pred`). This correlation, that would be $R = 0.785$. 

We can illustrate this with a scatter plot, with the predicted strength in the horizontal axis and the actual strength in the vertical axis. We build the graphic using the **Matplotlib pyplot API** (you can get a less ornamented version directly in Pandas, with the method `.plot.scatter()`).

```
In [17]: from matplotlib import pyplot as plt
```

```
In [18]: plt.figure(figsize=(6,6))
    ...: plt.title('Figure 1. Actual strength vs predicted strength')
    ...: plt.scatter(y_pred, y, color='black', s=2)
    ...: plt.xlabel('Predicted strength (MPa)')
    ...: plt.ylabel('Actual strength (MPa)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_2.1.png)

This visualization helps, in many cases, to detect whether something does not work as expected. In this case, we see that, in spite of the strong correlation, some of prediction errors are quite big. You should not be surprised. The correlation tells you that *on average* the errors are small, which is not the same as saying that all of them are small. Also, by focusing on vertical slices of this scatter plot, we see that, for a given predicted strength, the dispersion of the actual strength is bigger when the predicted strength is bigger. This is also a general fact.

## Q6. Save the model for future use

Finally, we can save our model to a **PKL file**. We import the package `joblib`:

```
In [19]: import joblib
```

Next, we apply the `joblib` function `dump` to create the PKL file:

```
In [20]: joblib.dump(reg, 'reg.pkl')
Out[20]: ['reg.pkl']
```

Now we have new file in the working directory (in Jupyter apps, you can learn where is that with the magic command `%pwd`. If you prefer to put the PKL file in a different place, youy can do it by specifying the appropriate path. You can recover the model, anytime, even if you no longer have the training data, as:

```
In [21]: newreg = joblib.load('reg.pkl')
```

`newreg` is like a copy of `reg`. Indeed, both models give the same predictions:

```
In [22]: (reg.predict(X) != newreg.predict(X)).sum()
Out[22]: 0
```
