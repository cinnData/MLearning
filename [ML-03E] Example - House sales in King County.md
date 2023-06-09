# [ML-03E] Example - House sales in King County

## Introduction

This example illustrates linear regression in scikit-learn. It shows how to develop a model for **house sale prices** in King County (Washington), which includes Seattle. King is the most populous county in Washington (population 1,931,249 in the 2010 census), and the 13th-most populous in the United States. The data include the homes sold between May 2014 and May 2015.

## The data set

The data come in the file `king.csv`. It contains 13 house features plus the sale price and date, along with 21,613 observations.

The variables are:

* `id`, an identifier of the house.

* `date`, the date when the sale took place.

* `zipcode`, the ZIP code of the house.

* `lat`, the latitude of the house.

* `long`, the longitude of the house.

* `bedrooms`, the number of bedrooms.

* `bathrooms`, the number of bathrooms.

* `sqft_above`, the square footage of the house, discounting the basement.

* `sqft_basement`, the square footage of the basement.

* `sqft_lot`, the square footage of the lot.

* `floors`, the total floors (levels) in house.

* `waterfront`, a dummy for having a view to the waterfront.

* `condition`, a 1-5 rating.

* `yr_built`, the year when the house was built.

* `yr_renovated`, the year when the house was renovated.

* `price`, the sale price.

Source: Kaggle.

## Questions

Q1. How is the distribution of the sale price?

Q2. Develop a linear regression model for predicting the sale price in terms of the house features, leaving aside the zipcode. Evaluate this model.

Q3. Plot the actual price versus the price predicted by the model. What do you see?

Q4. Add a dummy for every zipcode to the feature collection and run the analysis again. What happened?

# Importing the data

As in the preceding example, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `id` as the index (this is the role of the argument `index_col=0`.

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'king.csv', index_col=0)
````

## Exploring the data

`df` is a Pandas data frame. A report of the content can be printed with the method `.info()`. Everything is as expected, so far. There are no missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 21613 entries, 7129300520 to 1523300157
Data columns (total 15 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   date           21613 non-null  object 
 1   zipcode        21613 non-null  int64  
 2   lat            21613 non-null  float64
 3   long           21613 non-null  float64
 4   bedrooms       21613 non-null  int64  
 5   bathrooms      21613 non-null  float64
 6   sqft_above     21613 non-null  int64  
 7   sqft_basement  21613 non-null  int64  
 8   sqft_lot       21613 non-null  int64  
 9   floors         21613 non-null  float64
 10  waterfront     21613 non-null  int64  
 11  condition      21613 non-null  int64  
 12  yr_built       21613 non-null  int64  
 13  yr_renovated   21613 non-null  int64  
 14  price          21613 non-null  int64  
dtypes: float64(4), int64(10), object(1)
memory usage: 2.6+ MB
```

The index of this data frame, which we can manage as `df.index`, is the same as the column `id` of the original data. There are duplicates there:

```
In [3]: df.index.duplicated().sum()
Out[3]: 177
```

The logic of this calculation is as follows. The method `.duplicated()` returns a Boolean series signalling the duplicates (reading top-down, those values that have appeared before). With `.sum()`, we count the `True` values. With the same logic, but applying `.duplicated()` to the (columns of the) data frame, we can check that there are no duplicated rows.

```
In [4]: df.duplicated().sum()
Out[4]: 0
```

The potential explanation is that the same house could have been sold more than once during the period covered by the data. Indeed this what we find by focusing on the duplicated ID's.

```
In [5]: duplicates = df.index[df.index.duplicated()]
```

```
In [6]: df.loc[duplicates].head()
Out[6]: 
                       date  zipcode      lat     long  bedrooms  bathrooms   
id                                                                            
6021501535  20140725T000000    98117  47.6870 -122.386         3       1.50  \
6021501535  20141223T000000    98117  47.6870 -122.386         3       1.50   
4139480200  20140618T000000    98006  47.5503 -122.102         4       3.25   
4139480200  20141209T000000    98006  47.5503 -122.102         4       3.25   
7520000520  20140905T000000    98146  47.4957 -122.352         2       1.00   

            sqft_above  sqft_basement  sqft_lot  floors  waterfront   
id                                                                    
6021501535        1290            290      5000     1.0           0  \
6021501535        1290            290      5000     1.0           0   
4139480200        2690           1600     12103     1.0           0   
4139480200        2690           1600     12103     1.0           0   
7520000520         960            280     12092     1.0           0   

            condition  yr_built  yr_renovated    price  
id                                                      
6021501535          3      1939             0   430000  
6021501535          3      1939             0   700000  
4139480200          3      1997             0  1384000  
4139480200          3      1997             0  1400000  
7520000520          3      1922          1984   232000  
```

So, we leave the duplicates where they are. We rescale the sale price to the thousands, to have a cleaner picture. 

```
In [7]: df['price'] = df['price']/1000
```
Now, we go for the questions proposed.

## Q1. Distribution of the sale price

The distribution of a numeric series can be quickly explored in two ways. First, the method `.describe()` extracts a statistical summary. The maximum price suggests that we may have a long right tail, which can be expected in real estate prices.

```
In [8]: df['price'].describe()
Out[8]: 
count    21613.000000
mean       540.088142
std        367.127196
min         75.000000
25%        321.950000
50%        450.000000
75%        645.000000
max       7700.000000
Name: price, dtype: float64
```

Second, we can use a **histogram**. Histograms can be obtained directly in Pandas, but we prefer to use `matplotlib.pyplot` in this course, even if the specification gets longer, because it allows for better control of the graphical output. The histogram confirms our guess about the **skewness** of the distribution.

```
In [9]: from matplotlib import pyplot as plt
```

```
In [10]: plt.figure(figsize=(7,5))
    ...: plt.title('Figure 1. Actual price')
    ...: plt.hist(df['price'], color='gray', rwidth=0.97)
    ...: plt.xlabel('Sale price (thousands)');
```
![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_3.1.png)

## Q2. Linear regression equation

We are going to scikit-learn to obtain our regression models (not the only way in Python), so we create a target vector and a feature matrix. The target vector is the last column (`price`) and the feature matrix contains the other columns minus `date` and `zipcode`.

```
In [11]: y = df.iloc[:, -1]
    ...: X = df.iloc[:, 2:-1]
```

Alternatively, you can use the names of the columns, setting `y = df['strength']` and `X = df.drop(columns=['date', 'zipcode', 'price'])`. Now, we import the **estimator class** `LinearRegression` from the subpackage `linear_model`. We instantiate an estimator from this class, calling it `reg`, to reming us of the job. 

```
In [12]: from sklearn.linear_model import LinearRegression
    ...: reg = LinearRegression()
```

The method `.fit()` calculates the optimal equation. Since we are using the default of `LinearRegression`, which is **least squares** regression, the **loss function** is the MSE.

```
In [13]: reg.fit(X, y)
Out[13]: LinearRegression()
```

The predicted prices for the houses included the data set are then calculated with the method `.predict()`.

```
In [14]: y_pred = reg.predict(X)
```

Finally, we obtain a preliminary evaluation of the model with the method `.score()`.

```
In [15]: reg.score(X, y).round(3)
Out[15]: 0.646
```

This gives us a R-squared value of 0.646. Since this least squares regression, we can interpret it as a squared correlation. So, the correlation between actual prices (`y`) and predicted prices (`y_pred1`) is 0.804.

## Q3. Plot the actual price versus the price predicted by your model

We create this scatter plot also with `matplolibt.pyplot`. The argument `s=2` controls the size of the dots. One sets the size taking into account the number of samples, sometimes after a bit of trial and error.

```
In [16]: plt.figure(figsize=(5,5))
    ...: plt.title('Figure 2. Actual price vs predicted price')
    ...: plt.scatter(x=y_pred, y=y, color='black', s=2)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Actual price (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_3.2.png)

This type of visualization helps to understand the data, and to detect undesired effects. In this case, we see that, in spite of the strong correlation, the prediction error can be big. This could be expected, since the correlation only ensures an average predictive performance, and we have more than 20,000 samples.

Paying a bit more of attention, we can see that the biggest errors (in absolute value) happen in the most expensive houses. This also a well know fact: the bigger is what you measure, the bigger are the measurement errors. We can visualize the situation with a scatter plot.

```
In [17]: plt.figure(figsize=(5,5))
    ...: plt.title('Figure 3. Absolute prediction error vs predicted price')
    ...: plt.scatter(x=y_pred, y=abs(y-y_pred), color='black', s=1)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Absolute predicted error (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_3.3.png)

Another issue is that some of the predicted prices are negative. We can count them:

```
In [18]: (y_pred < 0).sum()
Out[18]: 38
```

This may look pathological to you, but it is not rare in this type of data. Since the average error is null (this is a property of least squares), we have, more or less, the same amount of positive and negative erros. When a cheap house has a negative and substantial error, the predicted price can be negative. A different thing is the isolated point that we observe on the left of the two above figures. Something is wrong in this sample.

## Q4. Dummies for the zipcodes

Since we are going to add the zipcode to the equation, we drop the longitude and the laitude, and pack the remaining features in a matrix:

```
In [19]: X1 = df.iloc[:, 4:-1]
```

To create the dummies, we use the Pandas function `get_dummies()`, which returns the dummies as the columns of a data frame. The data type is `bool`. 

```
In [20]: X2 = pd.get_dummies(df['zipcode'])
```

Now, `X2` has 70 columns (as many as different zipcodes in the data set). The column names are the zipcode values. The advantage of `get_dummies()`, versus an alternative system included in scikit-learn, is that the columns have names, so you know what is what. The drawback is that the names are numbers, and scikit-learn does not accept data frames with numeric names. We will fix this below.

```
In [21]: X2.head()
Out[21]: 
            98001  98002  98003  98004  98005  98006  98007  98008  98010   
id                                                                          
7129300520  False  False  False  False  False  False  False  False  False  \
6414100192  False  False  False  False  False  False  False  False  False   
5631500400  False  False  False  False  False  False  False  False  False   
2487200875  False  False  False  False  False  False  False  False  False   
1954400510  False  False  False  False  False  False  False  False  False   

            98011  ...  98146  98148  98155  98166  98168  98177  98178   
id                 ...                                                    
7129300520  False  ...  False  False  False  False  False  False   True  \
6414100192  False  ...  False  False  False  False  False  False  False   
5631500400  False  ...  False  False  False  False  False  False  False   
2487200875  False  ...  False  False  False  False  False  False  False   
1954400510  False  ...  False  False  False  False  False  False  False   

            98188  98198  98199  
id                               
7129300520  False  False  False  
6414100192  False  False  False  
5631500400  False  False  False  
2487200875  False  False  False  
1954400510  False  False  False  

[5 rows x 70 columns]
```

With Pandas function `concat()`, we join the two parts of the new feature matrix. The argument `axis=1` indicates that the two submatrices are joined horizontally (the default is to join vertically).
```
In [22]: X = pd.concat([X1, X2], axis=1)
```

Indeed, the new matrix has the right shape:

```
In [23]: X.shape
Out[23]: (21613, 80)
```

To prvent the trouble with the column names, we turn `X`into a NumPy 2D array:

```
In [24]: X = X.values
```

Now we fit the new data. This replaces the former model by a new one, which takes 80 features instead of 12. You could instantiate a new estimator with a different name, keeping both models alive.

```
In [25]: reg.fit(X, y)
Out[25]: LinearRegression()
```

The new predictions are:

```
In [26]: y_pred = reg.predict(X)
```

And the new R-squared value:

```
In [27]: reg.score(X, y).round(3)
Out[27]: 0.785
```

This looks like a relevant improvement, compared to the former model. The scatter plot shows the increased correlation:

```
In [28]: plt.figure(figsize=(5,5))
    ...: plt.title('Figure 4. Actual price vs predicted price')
    ...: plt.scatter(x=y_pred, y=y, color='black', s=2)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Actual price (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_3.4.png)

There are still negative predicted prices:

```
In [29]: (y_pred < 0).sum()
Out[29]: 16
```

## Homework

1. The role of longitude and latitude in the prediction of real estate prices is unclear. Do they really contribute to get better predictions in the first model of this example? If we keep them in the second model, do we get a better model? 

2. Evaluate in dollar terms the predictive performance of the two models presented in this example. For instance, you can use the mean (or median) absolute error.  Can you make a statement like "the value of x% of the houses can be predicted with an error below y thousand dollars"?

3. Is it better to use the percentage error in the above assessment?

4. Can the strong correlation be an artifact created by the extreme values? Trim the data set, dropping the houses beyond a certain threshold of price and/or size. Do you get a better model?

5. The distribution of the price is quite skewed, which is a fact of life in real state. The extreme values in the right tail of the distribution can exert an undesired influence on the regression coefficients. Transformations, such as the square root or the logarithm, are recommended in Statistics textbooks in many situations. In particular, the **log transformation** is recommended for variables with skewed distributions, to limit the influence of extreme values. Develop and evaluate a model for predicting the price which is based on a linear regression equation which has the logarithm of the price on the left side. 
