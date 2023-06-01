# [ML-03E] Example - House sales in King County

## Introduction

The objective of this example is to develop a model for **house sale prices** in King County (Washington), which includes Seattle. King is the most populous county in Washington (population 1,931,249 in the 2010 census), and the 13th-most populous in the United States. The data include the homes sold between May 2014 and May 2015.

## The data set

The data set comes in the file `king.csv`. It contains 13 house features plus the sale price and date, along with 21,613 observations.

The variables are:

* `id`, an identifier.

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

Q2. Develop a linear regression equation for predicting the sale price in terms of the available features. Evaluate this predictive model.

Q3. Plot the actual price versus the price predicted by your model. What do you see?

Q4. Add a collection of dummies asscoiated to the zipcode and run the analysis again. What happened?

# Importing the data

We use the Pandas funcion `read_csv()` to import the data. We use a remote path, to a GitHub repository. We take the column `id` as the index.

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'king.csv', index_col=0)
````

## Exploring the data

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

```
In [3]: df.index.duplicated().sum()
Out[3]: 177
```

```
In [4]: df.duplicated().sum()
Out[4]: 0
```

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

```
In [7]: df['price'] = df['price']/1000
```

## Q1. Distribution of the sale price

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

Histogram. Using `matplot.pyplot`.

```
In [9]: from matplotlib import pyplot as plt
```

```
In [10]: plt.figure(figsize=(8,6))
    ...: plt.title('Figure 1. Actual price')
    ...: plt.hist(df['price'], color='gray', rwidth=0.97)
    ...: plt.xlabel('Sale price (thousands)');
```
![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_3.1.png)

## Q2. Linear regression equation

Target vector and features matrix

```
In [11]: y = df.iloc[:, -1]
    ...: X = df.iloc[:, 2:-1]
```

Alternatively, you can use the names of the columns, setting `y = df['strength']` and `X = df.drop(columns=['date', 'zipcode', 'price'])`.

```
In [12]: from sklearn.linear_model import LinearRegression
    ...: model = LinearRegression()
```

```
In [13]: model.fit(X, y)
Out[13]: LinearRegression()
```

```
In [14]: y_pred = model.predict(X)
```

```
In [15]: round(model.score(X, y), 3)
Out[15]: 0.646
```

## Q3. Plot the actual price versus the price predicted by your model

Scatter plot. Actual price vs predicted price.


```
In [16]: plt.figure(figsize=(6,6))
    ...: plt.title('Figure 2. Actual price vs predicted price')
    ...: plt.scatter(y_pred, y, color='black', s=2)
    ...: plt.xlabel('Predicted price (thousands)')
    ...: plt.ylabel('Actual price (thousands)');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_3.2.png)

## Dummies for the zipcodes

```
In [17]: X1 = df.iloc[:, 4:-1]
```

```
In [18]: X2 = pd.get_dummies(df['zipcode'])
```

```
In [19]: X2.head()
Out[19]: 
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

```
In [20]: X = pd.concat([X1, X2], axis=1)
```

```
In [21]: X.shape
Out[21]: (21613, 80)
```

```
In [22]: X = X.values
```

```
In [23]: model.fit(X, y)
Out[23]: LinearRegression()
```

```
In [24]: y_pred = model.predict(X)
```

```
In [25]: round(model.score(X, y), 3)
Out[25]: 0.785
`
