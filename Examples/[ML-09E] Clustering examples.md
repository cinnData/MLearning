# [ML-09E] Clustering examples

## Introduction

This clustering exercise uses the data of the example **The spam filter** (`spam.csv`). The idea is to explore what an **unsupervised learning** approach, based on a **clustering algorithm**, would give for these data. Would the clusters obtained match the spam/ham split?

## Questions

The questions are:

Q1. Extract two **clusters** from the feature matrix, using the *k*-means method. Do these clusters match the 0/1 groups given by the target column `spam`?

Q2. Drop the three `cap_` variables and **binarize** all the `word_` variables, transforming them into dummies for the occurrence of the corresponding word. Repeat the analysis of question 1 with these binarized data.

Q3. Repeat the clustering exercise with the binarized data, after removing a few features, those that contribute less to predict spamness. Compare the results of the three analyses. What do you conclude?

## Importing the spam data

We import the data from the GitHub repository, as we did previously. We use the function `read_csv()` without any index specification. 

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'spam.csv')
```

We also separate the target vector and the feature matrix:

```
In [2]: y = df['spam']
   ...: X = df.drop(columns='spam')
```

## Q1. Two-cluster analysis (original data)

In scikit-learn, the protocol to work with unsupervised learning estimators is similar to that of the supervised learning examples that have already appeared in this course. We import the estimator class `KMeans()` from the subpackage `cluster`:

```
In [3]: from sklearn.cluster import KMeans
```

The, we instantiate an estimator, which we call `clus`, setting the number of clusters with the argument `n_clusters=2` and adding `random_state=0` to ensure the reproducibility (remember that the *k*-means algorith has a random start:

```
In [4]: clus = KMeans(n_clusters=2, random_state=0)
```

Next, we fit the estimator to the feature matrix `X`. Note that, in unsupervised learning, it is `.fit(X)`, instead of `.fit(X, y)`.

```
In [5]: clus.fit(X)
Out[5]: KMeans(n_clusters=2, random_state=0)
```

Once the estimator has been fitted, we can extract the attribute `.labels_`. With two clusters, the **labels** are 0 and 1. They come as a 1D array in which every term indicates the cluster to which the corresponding data unit has been assigned. I call this vector `label1`. We could add it as a new column to `df`, but we will keep it apart in this example.

```
In [6]: label1 = clus.labels_
   ...: label1
Out[6]: array([0, 0, 1, ..., 0, 0, 0])
```

To respond question Q1, we cross tabulate these labels with the target vector:

```
In [7]: pd.crosstab(y, label1)
Out[7]: 
col_0     0    1
spam            
0      2735   53
1      1622  191
```

The match is quite poor, and the reason is clear. The partition into two clusters is unbalanced. This is not strange when the clustering variables have mixed scales. Probably this would change after normalizing the feature matrix, but in this case we adopt a more radical approach, as suggested in question Q2.

Since we have more cases in the main diagonal ( 2,735 + 191) that in the secondary diagonal (1,622 + 53) If we wish to evaluate the match with a single number, we can calculate an "accuracy" as: 

```
In [8]: (y == label1).mean().round(3)
Out[8]: 0.636
```

With a spam rate of 39.4%, a result so close to 60% means practically nothing.

## Q2. Binary data set

We get the new feature matrix as:

```
In [9]: BX = (X.iloc[:, :-3] > 0).astype('int')
```

Note that, with `X.iloc[:, :-3]`, we exclude the last three columns. Then, `X.iloc[:, :-3] > 0` is a Boolean data frame. In the place `(i, j)` we have `True` when the *i*-th message contains the *j*-th word, and `false` otherwise. `.astype('int')` converts the Booleans to 0/1, just to follow the instructions, because this is not really needed for the exercise.

Now, we fit the estimator `clus` to the new training data `BX`, extracting the cluster labels as an array that we call `label2`.

```
In [10]: clus.fit(BX)
    ...: label2 = clus.labels_
```

Now, the cross tabulation produces something more promising.

```
In [11]: pd.crosstab(y, label2)
Out[11]: 
col_0     0     1
spam             
0       293  2495
1      1173   640
```

Note that the cluster labels mean nothing, so cluster 0 could be either spam or not spam. Here we would label cluster 0 as spam and cluster 1 and not spam. The accuracy of such assignation can be calcuated as:

```
In [12]: (y == 1 - label2).mean().round(3)
Out[12]: 0.797
```

This shows that unsupervised learning could be a first approximation to spam filtering, with no previous labeling by humans needed.

## Q3. Removing features

An easy way of selecting features is to train a decision tree classifier, using **feature importance** to rank the features. For the binary data set, this is can be done as follows.

```
In [13]: from sklearn.tree import DecisionTreeClassifier
    ...: clf = DecisionTreeClassifier(max_depth=5)
    ...: clf.fit(BX, y)
Out[13]: DecisionTreeClassifier(max_depth=5)
```

The feature importance vector for this decision tree classifier produces a first selection of 15 features.

```
In [14]: clf.feature_importances_
Out[14]: 
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

With `.iloc` slection, we can easily create a sub data frame containing only the selected columns.

```
In [15]: DBX = BX.iloc[:, clf.feature_importances_ > 0]
```

We fit the estimator `clus` to the new data, extracting the vector of labels:

```
In [16]: clus.fit(DBX)
    ...: label3 = clus.labels_
```

The new cross tabulation shows a better match.

```
In [17]: pd.crosstab(y, label3)
Out[17]: 
col_0     0     1
spam             
0       176  2612
1      1300   513
```

Labeling cluster 0 as spam, we would get a 85% accuracy.

```
In [18]: (y == 1 - label3).mean().round(3)
Out[18]: 0.85
```

## Homework

1. Create a feature matrix for the MNIST data (`digits.csv.zip`) and extract ten clusters from it, using the *k*-means method. 

2. Which is the digit that is better matched by one of the clusters? Can you assign a digit to every cluster?
