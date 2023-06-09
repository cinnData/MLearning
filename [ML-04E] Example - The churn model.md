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

Q1. Develop a model, based on logistic regression equation, to calculate a **churn score** for each customer.

Q2. How is the distribution of churn scores? Is it different for the churners and the non-churners?

Q3. Set an adequate **threshold** for the churn score and apply it to decide which customers are potential churners. What is the **true positive rate**? And the **false positive rate**?

# Importing the data

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

## Q2. Distribution of the churn scores


## Q3. Set a threshold for the churn score
