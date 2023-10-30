# [ML-09E] Clustering examples

## Spam data

Our first clustering exercise uses the data of the example **The spam filter** (`spam.csv`). The questions are:

Q1. Extract two **clusters** from the feature matrix, using the *k*-means method. Do these clusters match the 0/1 groups given by the target column `spam`?

Q2. Drop the three `cap_` variables and **binarize** all the `word_` variables, transforming them into dummies for the occurrence of the corresponding word. Repeat the analysis of question 1 with these binarized data.

Q3. Repeat the clustering exercise with the binarized data, after removing a few features, those that contribute less to predict spamness. Compare the results of the three analyses. What do you conclude?

## Importing the spam data

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'spam.csv')
```

```
In [2]: y = df['spam']
   ...: X = df.drop(columns='spam')
```

## Q1. 2-cluster analysis (original data)

```
In [3]: from sklearn.cluster import KMeans
   ...: clus = KMeans(n_clusters=2, random_state=0)
   ...: clus.fit(X)
Out[3]: KMeans(n_clusters=2, random_state=0)
```

```
In [4]: label1 = clus.labels_
   ...: label1
Out[4]: array([0, 0, 1, ..., 0, 0, 0])
```

```
In [5]: pd.crosstab(y, label1)
Out[5]: 
col_0     0    1
spam            
0      2735   53
1      1622  191
```

```
In [6]: (y == label1).mean().round(3)
Out[6]: 0.636
```

## Q2. Binary data set

```
In [7]: BX = (X.iloc[:, :-3] > 0).astype('int')
```

```
In [8]: clus.fit(BX)
   ...: label2 = clus.labels_
```

```
In [9]: pd.crosstab(y, label2)
Out[9]: 
col_0     0     1
spam             
0       293  2495
1      1173   640
```

```
In [10]: (y == 1 - label2).mean().round(3)
Out[10]: 0.797
```

## Q3. Removing features

```
In [11]: from sklearn.tree import DecisionTreeClassifier
    ...: clf = DecisionTreeClassifier(max_depth=5)
    ...: clf.fit(BX, y)
Out[11]: DecisionTreeClassifier(max_depth=5)
```

```
In [12]: clf.feature_importances_
Out[12]: 
array([0.        , 0.00267285, 0.        , 0.        , 0.00094443,
       0.        , 0.42095175, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.22679459, 0.00135792, 0.        , 0.        , 0.        ,
       0.00105616, 0.        , 0.04219654, 0.10378046, 0.07920994,
       0.        , 0.04232422, 0.00506278, 0.        , 0.00287113,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.02414783, 0.        , 0.        , 0.        ,
       0.        , 0.0008169 , 0.        , 0.        , 0.        ,
       0.04581248, 0.        , 0.        ])
```

```
In [13]: DBX = BX.iloc[:, clf.feature_importances_ > 0]
```

```
In [14]: clus.fit(DBX)
    ...: label3 = clus.labels_
```

```
In [15]: pd.crosstab(y, label3)
Out[15]: 
col_0     0     1
spam             
0       176  2612
1      1300   513
```

```
In [16]: (y == 1 - label3).mean().round(3)
Out[16]: 0.85
```

## MNIST data

Our second clustering exercise uses the MNIST data (`digits.csv.zip`). The questions are:

Q4. Extract ten clusters from the feature matrix, using the *k*-means method. 

Q5. Which is the digit that is better matched by one of the clusters? Can you assign a digit to every cluster?

## Importing the MNIST data

```
In [17]: df = pd.read_csv(path + 'digits.csv.zip')
    ...: y = df.iloc[:, 0]
    ...: X = df.iloc[:, 1:]
```

## Q4. 10-cluster analysis

```
In [18]: clus = KMeans(n_clusters=10, random_state=0)
    ...: clus.fit(X)
    ...: cluster = clus.labels_
```

```
In [19]: cluster
Out[19]: array([3, 9, 7, ..., 1, 2, 6])
```

## Q5. Best matches

```
In [20]: conf = pd.crosstab(y, cluster)
    ...: conf
Out[20]: 
col_0     0     1     2     3     4     5     6     7     8     9
label                                                            
0         9     7  1265    72     2   290   162    39     4  5053
1        10    11     7     8  4293     8     7     7  3526     0
2      4863    78   246   201   423   323   147   216   436    57
3       215    45   462  1083   449  4581    31   193    58    24
4        29  2173   288    17   178     0   168  3728   234     9
5         7   215  1811  1157   155  2129    67   432   280    60
6        53     4  2070    14   190    38  4324    67    45    71
7        53  4399    12    18   372     6     4  2094   314    21
8        53   194   291  4115   335  1212    51   208   330    36
9        19  2849    31    87   261    87    16  3462    95    51
```

```
In [21]: conf.max(axis=1)/conf.sum(axis=1)
Out[21]: 
label
0    0.732001
1    0.545004
2    0.695708
3    0.641507
4    0.546307
5    0.337241
6    0.628854
7    0.603181
8    0.602930
9    0.497557
dtype: float64
```

```
In [25]: conf.max(axis=0)/conf.sum(axis=0)
Out[25]: 
col_0
0    0.915647
1    0.441003
2    0.319297
3    0.607649
4    0.644788
5    0.528130
6    0.868796
7    0.356883
8    0.662533
9    0.938870
dtype: float64
```
