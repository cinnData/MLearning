# [ML-04E] Example - The churn model

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

As in the preceding example, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `id` as the index (this is the role of the argument `index_col=0`.

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'churn.csv', index_col=0)
````

## Exploring the data

`df` is a Pandas data frame. A report of the content can be printed with the method `.info()`. Everything is as expected, so far. There are no missing values.

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

The index of this data frame, which we can manage as `df.index`, is the same as the column `id` of the original data. There are no dupicate phone numbers, nor two customers with the same data:

```
In [3]: df.index.duplicated().sum() + df.duplicated().sum()
Out[3]: 0
```

## Q1. Logistic regression equation

We use scikit-learn to obtain our logistic regression model (not the only way in Python), so we create a target vector and a feature matrix. The target vector is the last column (`churn`) and the feature matrix contains the other columns.

```
In [4]: y = df['churn']
   ...: X = df.drop(columns='churn')
```

Alternatively, you can `.iloc` specifications here. Now, we import the **estimator class** `LogisticRegression` from the subpackage `linear_model`. We instantiate an estimator from this class, calling it `clf`, to reming us of the job. Instead of accepting the default arguments, as we did in the linear regression example, we increase the maximum number of interactions, whose default is 100, to 1,500. We leave the discussion of this point for the homework.

```
In [5]: from sklearn.linear_model import LogisticRegression
   ...: clf = LogisticRegression(max_iter=1500)
```

The method `fit` works as in linear regression.

```
In [6]: clf.fit(X, y)
Out[6]: LogisticRegression(max_iter=1500)
```

For a classification model, the method `.score()` returns the **accuracy**, which is the proportion of right prediction:

```
In [7]: clf.score(X, y).round(3)
Out[7]: 0.842
```

84.2% of rigth prediction may look like an achievement, but it is not, since the data show **class imbalance**. With only 19.4% positive cases, 80.6% accuracy can be obtained in a trivial way. So let us take a closer look at what at the performance of this model.

As given by the method `.predict(), the `**predicted target values** are obtained as follows:

* A **class probability** is calculated for each target value. In this example, this means two complementary probabilities, one for churning (`y == 1`) and one for not churning (`y == 0`). These probabilities can be extracted with the method `.predict_proba()`.

* For every sample, the predicted target value is the one with higher probability.

For binary classification, this can also be described in terms of **predictive scores** and **threshold values**: the predicted target value is positive when the score exceeds the threshold and negative otherwise. In this case, the scores are extracted, as a 1D array, with:

```
In [8]: df['score'] = clf.predict_proba(X)[:, 1]
```

Mind that Python orders the target values alphabetically. This means that the negative class comes first. So, to get the scores, which are the probabilities of the positive class, I have selected the second column of the 2D array returned by the method `clf.predict_proba()`. Also, note that we have added the scores as a column to our data set, which is just an option, since we can manage it as a separate vector.

## Q2. Distribution of the churn scores

We can take a look at the distribution of the predictive scores through a histogram. In this case, I am going to plot separately the scores for the churners (968) and the non-churners (4032) groups.

We import `matplotlib.pyplot` as usual:

```
In [9]: from matplotlib import pyplot as plt
```

You can find below the code to plot the two histograms, side-by-side. This is a bit more advanced than what we have previously done in this course. The `plt.figure()` line specifies the total size of the figure. Then, `plt.subplot(1, 2, 1)` and `plt.subplot(1, 2, 2)` start the two parts of the code chunk, one for each subplot. These parts are easy to read after our previous experience with histograms. The argument `range=80,1)` is used to get intervals of length 0.1, which are easier to read. The argument `edgecolor=white` improves the picture. The default is `edgecolor=none`

Note that `plt.subplot(1, 2, i)` refers to the $i$-th subplot in a grid of one row and two columns. The subplots are ordered by row, from left to righ and from top to bottom.

```
In [10]: # Set the size of the figure
    ...: plt.figure(figsize = (12,5))
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

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_4e.1.png)

You can now imagine the cutoff as a vertical line, and move it, right or left of the default threshold 0.5. Samples falling on the right of the vertical line would be classified as positive. Those falling on the left, as negative.

## Q3. Set a threshold for the churn scores

The default threshold, used by the method `.predict()`, is 0.5. So, the predicted values for this threshold are obtained as:

```
In [11]: y_pred = clf.predict(X)
```

It is plainly seen, in Figure 2.a, that with this threshold we are missing more than one half of the churners. So, in spite of its accuracy, our model would not be adequate for the actual business application. 

The **confusion matrix** resulting from the cross tabulation of the actual and the predicted target values, will confirm this visual intuition. Confusion matrices can be obtained in many ways. For instance, with the function `confusion_matrix` of the scikit-learn subpackage `metrics`:

```
In [12]: from sklearn.metrics import confusion_matrix
    ...: confusion_matrix(y, y_pred)
Out[12]: 
array([[3897,  135],
       [ 656,  312]], dtype=int64)
```

Alternatively, this matrix could be obtained with the Pandas function `crosstab()`. Note that scikit-learn returns the confusion matrix as a NumPy 2D array, while Pandas would have returned it as a Pandas data frame. As we guessed from the histogram, our churn model is not capturing enough churners (304/968) for a business application. Let us try a different one.

Note that the accuracy returned by the method `.score()` is the sum of the diagonal terms of this matrix divided by the sum of all terms of the matrix. It can also be calculated directly:

```
In [13]: (y == y_pred).mean().round(3)
Out[13]: 0.842
```

To predict more positives, we have to lower the threshold. Figure 2.a suggests that we have to go down to about 0.2 to make a real difference, while Figure 2.b warns us against lowering it further. So, let us try 0.2. The new vector of predicted target values is then obtained as:

```
In [14]: y_pred = (df['score'] > 0.2).astype(int)
```

The new confusion matrix is:

```
In [15]: confusion_matrix(y, y_pred)
Out[15]: 
array([[3164,  868],
       [ 343,  625]], dtype=int64)
```

Indeed, we are capturing about 2/3 of the churners. This comes at the price of raising the false positives to 866, which affects the accuracy:

```
In [16]: (y == y_pred).mean().round(3)
Out[16]: 0.758
```

A clear way to summarize the evaluation of the model comes through the true positive and false positive rates. They can be extracted from the confusion matrix or calculated directly. The **true positive rate** is the proportion of true positives among the actual positives:

```
In [17]: y_pred[y == 1].mean().round(3)
Out[17]: 0.646
```

The **false positive rate** is the proportion of false positives among the actual negatives:

```
In [18]: y_pred[y == 0].mean().round(3)
Out[18]: 0.215
``` 

## Homework

1. There is no formula to calculate the coefficients of a logistic regression, as it happens with linear regression. So, the best thing you can get is an approximation to the optimal coefficients by means of an iterative method, which is called the **solver**. There are many options for the solver, and we have used here the scikit-learn default, to make it simple. But it turns out that the default number of iterations is not enough in many cases and you get a warning from Python. To grasp this point, try different values for the parameter `max_iter` in the specification of the logistic regression estimator. Examine how the number of iterations affects the accuracy of the classifier.

2. Assume that the Omicron management plans to offer a 20% discount to the customers that the model classifies as potential churners, and that this offer is going to have a 100% success, so the company will retain all the churners detected. Evaluate the benefit produced by this **retention policy** with the two models presented here.

3. Define a Python function which gives the benefit in terms of the threshold and find an optimal threshold for this retention policy.
