# [ML-02E] Example - Modeling the strength of concrete

## Introduction

In the field of engineering, it is crucial to have accurate estimates of the performance of building materials. These estimates are required in order to develop safety guidelines governing the materials used in the construction of buildings, bridges, and roadways.

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

## Importing the data

We use the Pandas funcion `read_csv()` to import the data. First, we import the package:

```
In [1]: import pandas as pd
```

We use a remote path, to a GitHub repository. None of the columns is used as the index.

```
In [2]: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'concrete.csv')
````

## Exploring the data

We take a look at the data with the stanadrd Pandas methods `.info()`, `.head()` and `.describe()`. The first one prints a report of the dara frame content. We can see that all the columns are numeric, and that there are no missing values.

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

The method `.head()` prints the first five rows.

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

Finally, the method `.describe()` prints a statistical summary. 

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

## Target vector and features matrix

Looking at this data set with a supervised learning perspective, we specify the concrete strength as the **target vector**, that is, what we wish to predict. To be consistent, we will always denote this vector (it could either a NumPy 1D array or a Pandas series) as `y`. In this case, the target vector is the last column, wg¡hich we can extract as:

```
In [6]: y = df.iloc[:, -1]
```

Alternatively, you can use the column name, setting `y = df['strength']`. The feature matrix packs the rest of the columns:

```
In [7]: X = df.iloc[:, :-1]
```

The same can be obtained as `X = df.drop(columns='strength')`.

## Linear regression model

To obtain a **linear regression model** for this example, we use the scikit-learn claa `LinearRegression`, from the subpackage `linear_model`. We create an instance of this estimator with:

```
In [8]: from sklearn.linear_model import LinearRegression
   ...: model = LinearRegression()
```

We apply next the three basic methods `.fit()`, `.predict()` and `.score()`. The first one returns the "model", that is an object that, given the components of the mixture, returns the predicted strength. In this case, the model consists in a linear equation, so what `.fit()` does is to calculate the coefficients for the equation. These coefficients are optimal (minimum MSE) for the data provided (the pair `X`, `y`).

```
In [9]: model.fit(X, y)
Out[9]: LinearRegression()
```



```
In [10]: y_pred = model.predict(X)
```

```
In [11]: round(model.score(X, y), 3)
Out[11]: 0.616
```

## Scatter plot

```
In [12]: from matplotlib import pyplot as plt
```

```
In [13]: plt.figure(figsize=(6,6))
    ...: plt.title('Figure 1. Actual strength vs predicted strength')
    ...: plt.scatter(y_pred, y, color='black', s=2)
    ...: plt.xlabel('Predicted strength')
    ...: plt.ylabel('Actual strength');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_2.1.png)

