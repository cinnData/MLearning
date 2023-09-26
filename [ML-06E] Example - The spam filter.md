# [ML-06E] Example - The spam filter

## Introduction

A **spam filter** is an algorithm which classifies e-mail messages as either spam or non-spam, based on a collection of **numeric features** such as the frequency of certain words or characters. In a spam filter, the **false positive rate**, that is, the proportion of non-spam messages wrongly classified as spam, must be very low.

## The data set

The file `spam.csv` contains data on 4,601 e-mail messages. Among these messages, 1,813 have been classified as spam. The data were gathered at Hewlett-Packard by merging: (a) a collection of spam e-mail from the company postmaster and the individuals who had filed spam, and (b) a collection of non-spam e-mail, extracted from filed work and personal e-mail.

The variables are:

* 48 numeric features whose names start with `word_`, followed by a word. They indicate the frequency, in percentage scale, with which that word appears in the message. Example: for a particular message, a value 0.21 for `word_make` means that 0.21% of the words in the message match the word 'make'.

* 3 numeric features indicating, respectively, the average length of uninterrupted sequences of capital letters (`cap_ave`), the length of the longest uninterrupted sequence of capital letters (`cap_long`) and the total number of capital letters in the message (`cap_total`).

* A dummy indicating whether that e-mail message is spam (`spam`).

Source: Hewlett-Packard. Taken from T Hastie, R Tibshirani & JH Friedman (2001), *The Elements of Statistical Learning*, Springer.

## Questions

Q1. Develop a spam filter based on a **decision tree** classifier with **maximum depth** 2 and evaluate its performance. 

Q2. Repeat the exercise allowing a maximum depth of 3.

Q3. Repeat the exercise allowing a maximum depth of 4.

Q4. Repeat the exercise allowing a maximum depth of 5.

Q5. Among the features available in this data set, which ones are more relevant for filtering spam?

## Importing the data

As in the preceding examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. Since the email messages don't have an identifier, we leave Pandas to create a `RangeIndex`. 

```
In [1]: import pandas as pd, numpy as np
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'spam.csv')
```

## Exploring the data

The dimensions of the data set are those expected from the description presented above.

```
In [2]: df.shape
Out[2]: (4601, 52)
```

We also take a look at the first rows (of some columns),with the methos `.head()`. 

```
In [3]: df.head()
Out[3]: 
   word_make  word_address  word_all  word_3d  word_our  word_over  \
0       0.00          0.64      0.64      0.0      0.32       0.00   
1       0.21          0.28      0.50      0.0      0.14       0.28   
2       0.06          0.00      0.71      0.0      1.23       0.19   
3       0.00          0.00      0.00      0.0      0.63       0.00   
4       0.00          0.00      0.00      0.0      0.63       0.00   

   word_remove  word_internet  word_order  word_mail  ...  word_original  \
0         0.00           0.00        0.00       0.00  ...           0.00   
1         0.21           0.07        0.00       0.94  ...           0.00   
2         0.19           0.12        0.64       0.25  ...           0.12   
3         0.31           0.63        0.31       0.63  ...           0.00   
4         0.31           0.63        0.31       0.63  ...           0.00   

   word_project  word_re  word_edu  word_table  word_conference  cap_ave  \
0           0.0     0.00      0.00         0.0              0.0    3.756   
1           0.0     0.00      0.00         0.0              0.0    5.114   
2           0.0     0.06      0.06         0.0              0.0    9.821   
3           0.0     0.00      0.00         0.0              0.0    3.537   
4           0.0     0.00      0.00         0.0              0.0    3.537   

   cap_long  cap_total  spam  
0        61        278     1  
1       101       1028     1  
2       485       2259     1  
3        40        191     1  
4        40        191     1  

[5 rows x 52 columns]
```
Everything looks right. We also check the **spam rate** in this data set, which agrees with the description (1,813/4,601).

```
In [4]: round(df['spam'].mean(), 3)
Out[4]: 0.394
```

## Target vector and feature matrix

We use scikit-learn to obtain our decision tree models, so we create a **target vector** and a **feature matrix**. The target vector is the last column (`spam`) and the feature matrix contains the other columns.

```
In [5]: y = df['spam']
   ...: X = df.drop(columns='spam')
```

## Q1. Decision tree classifier (max depth = 2)

To develop a tree decision classifier, we use the **estimator class** `DecisionTreeClassifier` from the scikit-learn subpackage `tree`. This is imported as usual:

```
In [6]: from sklearn.tree import DecisionTreeClassifier
```

We instantiate a first estimator from this class. We set `max_depth=2`, which limits the **depth**, that is, the length of the longest branch.

```
In [7]: clf1 = DecisionTreeClassifier(max_depth=2)
```

The method `.fit()` finds the optimal tree under this specification. This tree can be seen at the lecture ML-X. I has four leaves and uses only three of the 51 features available.

```
In [8]: clf1.fit(X, y)
Out[8]: DecisionTreeClassifier(max_depth=2)
```

The method `.score()` gives us the **accuracy** of this model.

```
In [9]: round(clf1.score(X, y), 3)
Out[9]: 0.834
```

83.4% accuracy looks promising for a spam filter, but we have been warned about the false positive rate. Also, we know that the training data have been gathered by mixing spam and legal mail in arbitrary proportions. Therefore, this 83.4% is not real, is just a weighted average of the accuracy of the model with spam mail and the accuracy with legal mail, but with arbitray weights. 

So, we take a closer look at the predictions of this model, y means of the **confusion matrix**. First, we extract a vector with the predicted spamness, with the method `.predict()`.

```
In [10]: y_pred1 = clf1.predict(X)
```

We obtain the confusion matrix as in other examples, with the function `confusion_matrix()`, from the scikit-learn subpackage `sklearn.metrics`.

```
In [11]: from sklearn.metrics import confusion_matrix
    ...: conf1 = confusion_matrix(y, y_pred1)
    ...: conf1
Out[11]: 
array([[2575,  213],
       [ 549, 1264]])
```

The counts in first row of this matrix correspond to the negative samples (the legal mail), while those in the second row corresponds to the positive samples (the spam mail). Here, the accuracy is poorer for the spam mail. To make the discussion more specific, we can use standard metrics like the true positive rate and the false positive rate.

```
In [12]: tp1 = conf1[1, 1]/sum(conf1[1, :])
    ...: fp1 = conf1[0, 1]/np.sum(conf1[0, :])
    ...: round(tp1, 3), round(fp1, 3)
Out[12]: (0.697, 0.076)
```

We would like to improve these statistics. Since the model we are using is supersimple, an obvious approach is to increase the maximum depth parameter. Note that a decision tree with depth 2 uses at most three features.

## Q2. Decision tree classifier (max depth = 3)

We instantiate a second decision tree classifier, `clf2`, with `max_depth=3`. 

```
In [13]: clf2 = DecisionTreeClassifier(max_depth=3)
    ...: clf2.fit(X, y)
Out[13]: DecisionTreeClassifier(max_depth=3)
```

Right now, we have two `DecisionTreeClassifier` estimators, `clf1` and `clf2`, both fitted to our training data. Alternatively, we could have continued with `clf1`, setting `clf1.max_depth = 3`. But, mind that every time we refit a model, the previous results are wiped off.

The confusion matrix gets better, specially the false negatives.

```
In [14]: y_pred2 = clf2.predict(X)
    ...: conf2 = confusion_matrix(y, y_pred2)
    ...: conf2
Out[14]: 
array([[2598,  190],
       [ 406, 1407]])
```

The true positive rate and the false positive rate are, now:

```
In [15]: tp2 = conf2[1, 1]/sum(conf2[1, :])
    ...: fp2 = conf2[0, 1]/sum(conf2[0, :])
    ...: round(tp2, 3), round(fp2, 3)
Out[15]: (0.776, 0.068)
```

At most, we have added four decision nodes, so this decision tree may be using seven features, out of the fifty-odd available features. To address question Q3, we allow the tree some extra growth.

## Q3. Decision tree classifier (max depth = 4)

We fit a new decision tree classifier, with `max_depth=4`.

```
In [16]: clf3 = DecisionTreeClassifier(max_depth=4)
    ...: clf3.fit(X, y)
Out[16]: DecisionTreeClassifier(max_depth=4)
```

Again, we examine the confusion matrix, which keeps improving. 

```
In [17]: y_pred3 = clf3.predict(X)
    ...: conf3 = confusion_matrix(y, y_pred3)
    ...: conf3
Out[17]: 
array([[2627,  161],
       [ 341, 1472]])
```

The true positive rate and the false positive rate improve accordingly. This tree may be using 15 features.

```
In [18]: tp3 = conf3[1, 1]/sum(conf3[1, :])
    ...: fp3 = conf3[0, 1]/sum(conf3[0, :])
    ...: round(tp3, 3), round(fp3, 3)
Out[18]: (0.812, 0.058)
```

## Q4. Decision tree classifier (max depth = 5)

In a final assault, we fit a new decision tree classifier, with `max_depth=5`.

```
In [19]: clf4 = DecisionTreeClassifier(max_depth=5)
    ...: clf4.fit(X, y)
Out[19]: DecisionTreeClassifier(max_depth=5)
```

The confusion matrix still improves on the false positive side.

```
In [20]: y_pred4 = clf4.predict(X)
    ...: conf4 = confusion_matrix(y, y_pred4)
    ...: conf4
Out[20]: 
array([[2696,   92],
       [ 350, 1463]])
```

We find an interesting decrease in the number of false positives, which is relevant for his example. The false positive rate confirms this.

```
In [21]: tp4 = conf4[1, 1]/sum(conf4[1, :])
    ...: fp4 = conf4[0, 1]/sum(conf4[0, :])
    ...: round(tp4, 3), round(fp4, 3)
Out[21]: (0.807, 0.033)
```

We stop here, since the model is getting complex, raising a concern about **overfitting**. We will deal with that in the forthcoming lectures.

## Q5. Feature relevance

One of the most attractive traits of decision trees is that they produce, as a by-product, a ranking of the features by their contribution to the predictive power of the model. In scikit-learn, this is the attribute `.feature_importances_` of the `DecisionTreeClassifier` estimator, extracted as a 1D array in which every term is the **importance** of one of the features. The importance is measured as the **percentage of reduction of the impurity** due to the splits in which the feature is involved. Zero importance means that the corresponding feature is not involved in any split, so it is not used by the decision tree.

We take a look at feature importance in the biggest of our trees. In spite of the allowance for depth 5, only 15 features are involved in the splits. 

```
n [22]: imp = clf4.feature_importances_
    ...: imp
Out[22]: 
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.40565322, 0.00866955, 0.        , 0.00535017,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.23436546, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.05686147, 0.1077894 , 0.03860056,
       0.        , 0.04083541, 0.00327628, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.0013128 , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.00253777, 0.        ,
       0.03341536, 0.        , 0.        , 0.05426278, 0.00463312,
       0.00243665])
```

To prepare a better report, we transform this array into a Pandas series, with the feature names as the index.

```
In [23]: feat_list = pd.Series(imp, index=df.columns[:51])
```

Finally, we filter out the non-relevant features and sort them by relevance.

```
feat_list[imp > 0].sort_values(ascending=False).round(3)
Out[33]: 
word_remove        0.406
word_free          0.234
word_money         0.108
word_000           0.057
cap_ave            0.054
word_george        0.041
word_hp            0.039
word_edu           0.033
word_internet      0.009
word_mail          0.005
cap_long           0.005
word_650           0.003
word_project       0.003
cap_total          0.002
word_technology    0.001
dtype: float64
```
