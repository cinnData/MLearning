# [MLE-03] Example - The churn model

## Introduction

The term churn is used in marketing to refer to a customer leaving the company in favor of a competitor. Churning is a common concern of **Customer Relationship Management** (CRM). A key step in proactive churn management is to predict whether a customer is likely to churn, since an early detection of the potential churners helps to plan the retention campaigns.

This example deals with a churn model based on a **logistic regression equation**, for a company called *Omicron Mobile*, which provides mobile phone services. The data set is based on a random sample of 5,000 customers whose accounts were still alive by September 30, and have been monitored during the fourth quarter. 968 of those customers churned during the fourth quarter, a **churning rate** of 19.4%.

## The data set

The variables included in the data set (file `churn.csv`) are:

* `id`, a customer ID (the phone number).

* `aclentgh`, the number of days the account has been active at the beginning of the period monitored.

* `intplan`, a dummy for having an international plan.

* `dataplan`, a dummy for having a data plan.

* `ommin`, the total minutes call to any Omicron mobile phone number, voicemail or national landline.

* `omcall`, the total number of calls to any Omicron mobile phone number, voicemail or national landline.

* `otmin`, the total minutes call to other mobile networks.

* `otcall`, the total number of calls to other networks.

* `ngmin`, the total minutes call to nongeographic numbers. Nongeographic numbers, such as UK 0844 or 0871 numbers, are often helplines for organizations like banks, insurance companies, utilities and charities.

* `ngcall`, the total number of calls to nongeographic numbers.

* `imin`, the total minutes in international calls.

* `icall`, the total international calls.

* `cuscall`, the number of calls to customer service.

* `churn`, a dummy for churning.

All the data are from the third quarter except the last variable.

Source: MA Canela, I Alegre & A Ibarra (2019), *Quantitative Methods for Management*, Wiley.

## Questions

Q1. Develop a model, based on logistic regression equation, to calculate a **churn score**, that is, an estimate of the probability of churning, for each customer.

Q2. How is the distribution of churn scores? Is it different for the churners and the non-churners?

Q3. Set an adequate **threshold** for the churn score and apply it to decide which customers are potential churners. What is the **true positive rate**? And the **false positive rate**?

## Importing the data

As in the preceding example, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `id` as the index (this is the role of the argument `index_col=0`).

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'churn.csv', index_col=0)
````

## Exploring the data

`df` is a Pandas data frame. A report of the content can be printed with the method `.info()`. We don't find here anything unexpected. Also, there are no missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 5000 entries, 409-8978 to 444-8504
Data columns (total 13 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   aclength  5000 non-null   int64  
 1   intplan   5000 non-null   int64  
 2   dataplan  5000 non-null   int64  
 3   ommin     5000 non-null   float64
 4   omcall    5000 non-null   int64  
 5   otmin     5000 non-null   float64
 6   otcall    5000 non-null   int64  
 7   ngmin     5000 non-null   float64
 8   ngcall    5000 non-null   int64  
 9   imin      5000 non-null   float64
 10  icall     5000 non-null   int64  
 11  cuscall   5000 non-null   int64  
 12  churn     5000 non-null   int64  
dtypes: float64(4), int64(9)
memory usage: 546.9+ KB
```

The index of this data frame, which we can manage as `df.index`, is the same as the column `id` of the original data. There are no duplicate phone numbers, nor two customers with the same data:

```
In [3]: df.index.duplicated().sum() + df.duplicated().sum()
Out[3]: 0
```

## Q1. Logistic regression equation

We use scikit-learn to obtain our logistic regression model, so we create a target vector and a feature matrix. The target vector is the last column (`churn`) and the feature matrix contains the other columns.

```
In [4]: y = df['churn']
   ...: X = df.drop(columns='churn')
```

Alternatively, we could have used `.iloc` specifications here. Now, we import the **estimator class** `LogisticRegression()` from the subpackage `linear_model`. We instantiate an estimator from this class, calling it `clf`. Instead of accepting the default arguments, as we did in the preceding example, we increase the maximum number of iterations, whose default is 100, to 1,500. Using the default `max_iter=100` would have raised a warning indicating that the optimization process has not finished.

```
In [5]: from sklearn.linear_model import LogisticRegression
   ...: clf = LogisticRegression(max_iter=1500)
```

The method `.fit()` works as in linear regression.

```
In [6]: clf.fit(X, y)
Out[6]: LogisticRegression(max_iter=1500)
```

For a classification model, the method `.score()` returns the **accuracy**, which is the proportion of right prediction:

```
In [7]: clf.score(X, y).round(3)
Out[7]: 0.842
```

84.2% of rigth prediction may look like an achievement, but it is not so in this case, since the data show **class imbalance**. With only 19.4% positive cases, 80.6% accuracy can be obtained in a trivial way. So let us take a closer look at the performance of this model.

As given by the method `.predict(), the `**predicted target values** are obtained as follows:

* A **class probability** is calculated for each target value. In this example, this means two complementary probabilities, one for churning (`y == 1`) and one for not churning (`y == 0`). These probabilities can be extracted with the method `.predict_proba()`.

* For every sample, the predicted target value is the one with higher probability.

For a binary classifier, this can also be described in terms of **predictive scores** and **threshold values**: the predicted class is positive when the score exceeds the threshold and negative otherwise. In scikit-learn, teh predicted probabilities are extracted as:

```
In [8]: clf.predict_proba(X)
Out[8]: 
array([[0.95309927, 0.04690073],
       [0.96888113, 0.03111887],
       [0.72661291, 0.27338709],
       ...,
       [0.87123317, 0.12876683],
       [0.40292348, 0.59707652],
       [0.17271608, 0.82728392]])

```

Mind that Python orders the target values alphabetically. This means that the negative class comes first. The scores are extracted as:

```
In [9]: df['score'] = clf.predict_proba(X)[:, 1]
```

Note that we have added the scores as a column to our data set, which is just an option, since we can manage it as a separate vector. The actual class and the predictive score are now the last two columns in `df`.

```
In [10]: df[['churn', 'score']]
Out[10]: 
          churn     score
id                       
409-8978      0  0.046901
444-7077      0  0.031119
401-9132      0  0.273387
409-2971      0  0.131950
431-5175      0  0.068075
...         ...       ...
390-2408      0  0.573438
407-6398      0  0.267838
444-7620      1  0.128767
352-4885      1  0.597077
444-8504      1  0.827284

[5000 rows x 2 columns]
``` 

## Q2. Distribution of the churn scores

We can take a look at the distribution of the predictive scores through a histogram. In this case, we plot separately the scores for the churners (968) and the non-churners (4,032) groups.

We import `matplotlib.pyplot` as usual:

```
In [11]: from matplotlib import pyplot as plt
```

You can find below the code to plot the two histograms, side-by-side. The `plt.figure()` line specifies the total size of the figure. Then, `plt.subplot(1, 2, 1)` and `plt.subplot(1, 2, 2)` start the two parts of the code chunk, one for each subplot. These parts are easy to read after our previous experience with histograms. The argument `range=(0,1)` is used to get intervals of length 0.1 (the default is `bins=10`), which are easier to read. The argument `edgecolor=white` improves the picture. The default of `plt.hist()`  takes the same value for `color` and `edgecolor`.

Note that `plt.subplot(1, 2, i)` refers to the $i$-th subplot in a grid of one row and two columns. The subplots are ordered by row, from left to righ and from top to bottom.

```
In [12]: # Set the size of the figure
    ...: plt.figure(figsize=(12,5))
    ...: # First subplot
    ...: plt.subplot(1, 2, 1)
    ...: plt.hist(df['score'][y == 1], range=(0,1), color='gray', edgecolor='white')
    ...: plt.title('Figure 1.a. Scores (churners)')
    ...: plt.xlabel('Churn score')
    ...: # Second subplot
    ...: plt.subplot(1, 2, 2)
    ...: plt.hist(df['score'][y == 0], range=(0,1), color='gray', edgecolor='white')
    ...: plt.title('Figure 1.b. Scores (non-churners)')
    ...: plt.xlabel('Churn score');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-03.1.png)

You can now imagine the cutoff as a vertical line, and move it, right or left of the default threshold 0.5. Samples falling on the right of the vertical line would be classified as positive. Those falling on the left, as negative.

## Q3. Set a threshold for the churn scores

The default threshold, used by the method `.predict()`, is 0.5. The predicted class for this threshold can be obtained as:

```
In [13]: y_pred = clf.predict(X)
```

It is plainly seen, in Figure 1.a, that with this threshold we are missing more than one half of the churners. So, in spite of its accuracy, our model would not be adequate for the actual business application. 

The **confusion matrix** resulting from the cross tabulation of the actual and the predicted target values, will confirm this visual intuition. Confusion matrices can be obtained in many ways. For instance, with the function `confusion_matrix` of the scikit-learn subpackage `metrics`:

```
In [14]: from sklearn.metrics import confusion_matrix
    ...: confusion_matrix(y, y_pred)
Out[14]: 
array([[3897,  135],
       [ 656,  312]], dtype=int64)
```

Alternatively, this matrix could be obtained with the Pandas function `crosstab()`. Note that scikit-learn returns the confusion matrix as a NumPy 2D array, while Pandas would have returned it as a Pandas data frame. Also, note that the accuracy returned by the method `.score()` is the sum of the diagonal terms of this matrix divided by the sum of all terms of the matrix. It can also be calculated directly:

```
In [15]: (y == y_pred).mean().round(3)
Out[15]: 0.842
```

As we guessed from the histogram, our churn model is not capturing enough churners (304/968) for a business application. Let us try a different one. To predict more positives, we have to lower the threshold. Figure 1.a suggests that we have to go down to about 0.2 to make a real difference, while Figure 1.b warns us against lowering it further. So, let us try 0.2. The new vector of clases is then obtained as:

```
In [16]: y_pred = (df['score'] > 0.2).astype(int)
```

The new confusion matrix is:

```
In [17]: confusion_matrix(y, y_pred)
Out[17]: 
array([[3164,  868],
       [ 343,  625]], dtype=int64)
```

Indeed, we are capturing now about 2/3 of the churners. This comes at the price of raising the false positives to 866, which affects the accuracy:

```
In [18]: (y == y_pred).mean().round(3)
Out[18]: 0.758
```

A clear way to summarize the evaluation of the model comes through the true positive and false positive rates. They can be extracted from the confusion matrix or calculated directly. The **true positive rate** is the proportion of predicted positives among the actual positives:

```
In [19]: y_pred[y == 1].mean().round(3)
Out[19]: 0.646
```

The **false positive rate** is the proportion of predicted positives among the actual negatives:

```
In [20]: y_pred[y == 0].mean().round(3)
Out[20]: 0.215
``` 
