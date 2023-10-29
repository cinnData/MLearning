# [ML-05E] Example - Term deposits

## Introduction

There are two main approaches for companies to promote products and/or services: through mass campaigns, targeting general indiscriminate public, or through **direct marketing**, targeting a specific set of contacts. Nowadays, in a global competitive world, positive responses to mass campaigns are typically very low. Alternatively, direct marketing focus on targets that assumably will be keener to that specific product/service, making these campaigns more attractive, because of their efficiency. But the increasingly vast number of marketing campaigns has reduced their effect on the general public. Furthermore, economical pressures and competition has led marketing managers to invest on direct campaigns with a strict and rigorous selection of contacts.

Due to the internal competition and the current financial crisis, there are huge pressures for European banks to increase their financial assets. One strategy is to offer attractive **long-term deposit** applications with good interest rates, in particular by using direct marketing campaigns.

A Portuguese institution has been offering term deposits to its clients for the past two years, but in a way that the board finds disorganized and inefficient. It looks as if too many contacts were made, for the subscriptions obtained.

The bank has been using its own contact-center to carry out direct marketing campaigns. The telephone was the dominant marketing channel, although sometimes with an auxiliary use of the Internet online banking channel (*e.g*. by showing information to a specific targeted client). Furthermore, each campaign was managed in an integrated fashion and the results for all channels were outputted together.

The manager in charge of organizing the next campaign is expected to optimize the effort. His objective is to find a **predictive model**, based on data of the preceding campaign, which can explain the success of a contact, *i.e*. whether the client subscribes the deposit. Such model can increase the campaign's efficiency by identifying the main characteristics that affect success, helping in a better management of the available resources (*e.g*. human effort, phone calls and time) and the selection of a high quality and affordable set of potential clients. To be useful for the direct campaign, a predictive model should allow reducing the number of calls in a relevant way without loosing a relevant number of subscribers.

## The data set

The data for this lecture come from the previous phone campaign of the bank, which involved a total of 45,211 contacts. During that campaign, an attractive long-term deposit application, with good interest rates, was offered. The contacts led to 5,289 subscriptions, a 11.7% **conversion rate**.

The data set combines demographic data with data about the interaction of the client and the bank. Some of the categorical variables (the type of job, the marital status, etc) have been transformed into **dummy variables** (1/0) so they can be directly entered in an equation:

* The client's account number (`accnum`).

* The client's  age in years (`age`).

* The client's type of job (`job`). The values are 'admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'student', 'technician', 'unknown' and 'unemployed'. Converted to twelve dummies.

* The client's marital status (`marital`). The values are 'married', 'divorced' and 'single'. Converted to three dummies.

* The client's education level (`education`). The values are 'unknown', 'secondary', 'primary' and 'tertiary'. Converted to four dummies.

* Whether the client has credit in default (`default`). The values are 'yes' and 'no'. Converted to a dummy (1 for 'yes' and 0 for 'no').

* The client's average yearly balance in euros (`balance`).

* Whether the client has a housing loan (`housing`). The values are 'yes' and 'no'. Converted to a dummy (1 for 'yes' and 0 for 'no').

* Whether the client has a personal loan (`loan`). The values are 'yes' and 'no'. Converted to a dummy (1 for 'yes' and 0 for 'no').

* The usual communication channel with the client (`contact`). The values are 'unknown', 'telephone' and 'cellular'. Converted to three dummies.

* The duration of last contact with the client before the campaign in seconds (`duration`). 

* The number of days passed by after the client was last contacted from a previous campaign (`pdays`). The entry is -1 when the client has not been previously contacted.

* Number of contacts performed before this campaign and for this client (`previous`).

* Outcome of the previous marketing campaign wuth the client (`poutcome`). The values are 'unknown', 'other', 'failure' and 'success'. Converted to four dummies.

* Whether the client has subscribed a term deposit (`deposit`). The values are 'yes' and 'no'. Converted to a dummy (1 for 'yes' and 0 for 'no').

Source: S Moro, P Cortez & P Rita (2014), A data-driven approach to predict the success of bank telemarketing, *Decision Support Systems* **62**, 22-31.

## Questions

Q1. Develop a **logistic regression model** to predict the response to the campaign (`deposit`) from the other variables.

Q2. Use your model to assign, to every client, a **predictive score** for suscribing the deposit. How is the distribution of the subscription scores obtained? Is it different for the subscribers and the non-subscribers?

Q3. Set a **threshold** for the scores to adequate the model to your business purpose.

Q4. Based on your model, if we set a **target** of 4,000 subscriptions, how many calls would we need, to hit the target?

Q5. If we set a **budget** 10,000 calls, how will we select the clients to be called? How many subscriptions will we get?

## Importing the data

As in the preceding examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. In this case, we take the column `accnum` as the index (`index_col=0`).

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'deposit.csv', index_col=0)
```

We print next a report of the content of `df` with the method `.info()`. Everything is as expected, so far. No missing values.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 45211 entries, 2065031284 to 2086934257
Data columns (total 35 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   age                  45211 non-null  int64
 1   job_admin            45211 non-null  int64
 2   job_blue-collar      45211 non-null  int64
 3   job_entrepreneur     45211 non-null  int64
 4   job_housemaid        45211 non-null  int64
 5   job_management       45211 non-null  int64
 6   job_retired          45211 non-null  int64
 7   job_self-employed    45211 non-null  int64
 8   job_services         45211 non-null  int64
 9   job_student          45211 non-null  int64
 10  job_technician       45211 non-null  int64
 11  job_unemployed       45211 non-null  int64
 12  job_unknown          45211 non-null  int64
 13  marital_divorced     45211 non-null  int64
 14  marital_married      45211 non-null  int64
 15  marital_single       45211 non-null  int64
 16  education_primary    45211 non-null  int64
 17  education_secondary  45211 non-null  int64
 18  education_tertiary   45211 non-null  int64
 19  education_unknown    45211 non-null  int64
 20  default              45211 non-null  int64
 21  balance              45211 non-null  int64
 22  housing              45211 non-null  int64
 23  loan                 45211 non-null  int64
 24  channel_cellular     45211 non-null  int64
 25  channel_telephone    45211 non-null  int64
 26  channel_unknown      45211 non-null  int64
 27  duration             45211 non-null  int64
 28  pdays                45211 non-null  int64
 29  previous             45211 non-null  int64
 30  poutcome_failure     45211 non-null  int64
 31  poutcome_other       45211 non-null  int64
 32  poutcome_success     45211 non-null  int64
 33  poutcome_unknown     45211 non-null  int64
 34  deposit              45211 non-null  int64
dtypes: int64(35)
memory usage: 12.4 MB
```

We also display the first five rows:

```
In [3]: df.head()
Out[3]: 
            age  job_admin  job_blue-collar  job_entrepreneur  job_housemaid   
accnum                                                                         
2065031284   58          0                0                 0              0  \
2051283096   44          0                0                 0              0   
2029034586   33          0                0                 1              0   
2070859436   47          0                1                 0              0   
2098635102   33          0                0                 0              0   

            job_management  job_retired  job_self-employed  job_services   
accnum                                                                     
2065031284               1            0                  0             0  \
2051283096               0            0                  0             0   
2029034586               0            0                  0             0   
2070859436               0            0                  0             0   
2098635102               0            0                  0             0   

            job_student  ...  channel_telephone  channel_unknown  duration   
accnum                   ...                                                 
2065031284            0  ...                  0                1       261  \
2051283096            0  ...                  0                1       151   
2029034586            0  ...                  0                1        76   
2070859436            0  ...                  0                1        92   
2098635102            0  ...                  0                1       198   

            pdays  previous  poutcome_failure  poutcome_other   
accnum                                                          
2065031284     -1         0                 0               0  \
2051283096     -1         0                 0               0   
2029034586     -1         0                 0               0   
2070859436     -1         0                 0               0   
2098635102     -1         0                 0               0   

            poutcome_success  poutcome_unknown  deposit  
accnum                                                   
2065031284                 0                 1        0  
2051283096                 0                 1        0  
2029034586                 0                 1        0  
2070859436                 0                 1        0  
2098635102                 0                 1        0  

[5 rows x 35 columns]
```

Finally, we check the conversion rate:

```
In [4]: df['deposit'].mean().round(3)
Out[4]: 0.117
```

*Note*. The value -1 in the column `pdays` when a client has not been previously contacted may look strange, but it is irrelevant in this case, since this situation is covered by the dummy `poutcome_unknown`. It can be proved, bit of algebra that changing the value imputted to avoid having missing values in `pdays` produces an equation with exactly the same predicted values.

## Q1. Logistic regression model

We create a target vector and a feature matrix. The target vector is the last column (`deposit`) and the feature matrix contains the other columns.

```
In [5]: y = df['deposit']
   ...: X = df.drop(columns='deposit')
```

To develop our logistic regression model with scikit-learn, we instantiate an estimator from the class `LogisticRegression()`, from the subpackage `linear_model`, applying the method `.fit()` as in previous examples. We also increase here the maximum number of iterations. 

```
In [6]: from sklearn.linear_model import LogisticRegression
   ...: clf = LogisticRegression(max_iter=2000)
   ...: clf.fit(X, y)
Out[6]: LogisticRegression(max_iter=2000)
```

The default predictions and the corresponding confusion matrix are obtained as in the previous example.

```
In [7]: y_pred = clf.predict(X)
   ...: conf = pd.crosstab(y, y_pred)
   ...: conf
Out[7]: 
col_0        0     1
deposit             
0        38968   954
1         3570  1719
```

On one side, these results look fine, since, using this model, we will call only 2,673 clients, capturing 1,719 subscriptions (64.3% conversion rate). On the other side, we are missing 3,570 potential subscribers (70.9%). The total accuracy can be extracted from the con fusion matrix, or calculated directly:

```
In [8]: acc = (y == y_pred).mean().round(3)
```

The accuracies on the two groups are: 

```
In [9]: acc1 = y_pred[y == 1].mean().round(3)
   ...: acc0 = (1 - y_pred[y == 0]).mean().round(3)
```

We can print them together as:

```
In [10]: acc, acc1, acc0
Out[10]: (0.9, 0.325, 0.976)
```

So, we have very good accuracy on the negative group and a very poor accuracy on the positive group. This typical of imbalanced training data. Let us see how this may change if we use the predcitive scores. 

## Q2. Predictive scores

The predictive scores come as the second column of the 2D array retirned by the method `.predict_proba()`:

```
In [11]: df['score'] = clf.predict_proba(X)[:, 1]
```

Adding the scores as an additional column to `df` allows us to display the target values and the scores together:

```
In [12]: df[['deposit', 'score']]
Out[12]: 
            deposit     score
accnum                       
2065031284        0  0.020247
2051283096        0  0.012138
2029034586        0  0.003953
2070859436        0  0.008954
2098635102        0  0.050026
...             ...       ...
2027086314        1  0.699955
2028473156        1  0.344148
2026897134        1  0.992496
2091483260        0  0.194690
2086934257        0  0.261455
```

On average, the scores are correct, and they have a skewed distribution (nothing wrong with this).

```
In [13]: df['score'].describe()
Out[13]: 
count    45211.000000
mean         0.118695
std          0.179120
min          0.001720
25%          0.022626
50%          0.052807
75%          0.118823
max          1.000000
Name: score, dtype: float64
```

We can have a better view with separate histograms. The code has already been used in another example.

```
In [14]: from matplotlib import pyplot as plt
    ...: # Set the size of the figure
    ...: plt.figure(figsize = (12,5))
    ...: # First subplot
    ...: plt.subplot(1, 2, 1)
    ...: plt.hist(df['score'][y == 1], color='gray', edgecolor='white')
    ...: plt.title('Figure 1.a. Scores (subscribers)')
    ...: plt.xlabel('Subscription score')
    ...: # Second subplot
    ...: plt.subplot(1, 2, 2)
    ...: plt.hist(df['score'][y == 0], color='gray', edgecolor='white')
    ...: plt.title('Figure 1.b. Scores (non-subscribers)')
    ...: plt.xlabel('Subscription score');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_5e.1.png)

This figure shows what is wrong with the threshold 0.5. In order to capture at least 2/3 of the potential subscribers, we have to set it within the range 10-20%.

## Q3. Set a threshold for the scores

Let us take the actual conversion rate (rounded) as the threshold.

```
In [15]: y_pred = (df['score'] > 0.11).astype(int)
```

The  new confusion matrix is:

```
In [16]: conf = pd.crosstab(y, y_pred)
    ...: conf
Out[16]: 
score        0     1
deposit             
0        32696  7226
1         1022  4267
```

We would call now 11,493 clients, capturing 4,267 subscribers (37.1% conversion rate), which is 4/5 of the potential subscribers. The accuracies are similar for the two groups:

```
In [17]: acc = (y == y_pred).mean().round(3)
    ...: acc1 = y_pred[y == 1].mean().round(3)
    ...: acc0 = (1 - y_pred[y == 0]).mean().round(3)
    ...: acc, acc1, acc0
Out[17]: (0.818, 0.807, 0.819)
```

## Q4. Target of 4,000 subscriptions

The manager can decide that he does not need to think on a threshold once he/she has the scores. He can set a target of a reasonable number of subscriptions and use the scores to selext the clients to be contacted. This can be managed easily in a spreadsheet, though we continue here our Python exercise.

The manager would sort the data by the scores, in descending order.

```
In [18]: df.sort_values('score', inplace=True, ascending=False)
    ...: df[['deposit', 'score']]
Out[18]: 
            deposit     score
accnum                       
2084617209        0  1.000000
2054970681        0  0.999984
2096318570        1  0.999983
2064903718        0  0.999963
2078910432        0  0.999958
...             ...       ...
2038624917        0  0.002003
2041538627        0  0.001960
2037089126        0  0.001908
2009467351        0  0.001746
2024137560        0  0.001720

[45211 rows x 2 columns]
```

Then, he/she would start contacting the top scored clients, until getting the desired 4,000 subscriptions. This can be managed by adding a column with the cumulative number of subscriptions, which can be created with the method `.cumsum()`.

```
In [19]: df['cum_subscription'] = df['deposit'].cumsum()
    ...: df[['deposit', 'score', 'cum_subscription']]
Out[19]: 
            deposit     score  cum_subscription
accnum                                         
2084617209        0  1.000000                 0
2096318570        1  0.999990                 1
2054970681        0  0.999987                 1
2064903718        0  0.999972                 1
2078910432        0  0.999970                 1
...             ...       ...               ...
2064928371        0  0.002513              5289
2038624917        0  0.002434              5289
2032496851        0  0.002398              5289
2024137560        0  0.002370              5289
2009467351        0  0.002308              5289

[45211 rows x 3 columns]
```

The first row where the column `cum_subscription` attains 40,000 will correspond to the last client contacted. In total, 9,788 clients would be contacted.

```
In [20]: (df['cum_subscription'] < 4000).sum() + 1
Out[20]: 9788
```

## Q5. Budget 10,000 calls

Suppose now that instead of setting an objective, the budget for the campaign allows for a certain number of contacts , for instance 10,000. The manager would pick now the first 10,000 rows of the data set. The account numbers are provided by the index labels:

```
In [21]: call_list = df.index[:10000]
    ...: call_list
Out[21]: 
Int64Index([2084617209, 2096318570, 2054970681, 2064903718, 2078910432,
            2041730862, 2071098526, 2037940512, 2032450716, 2095471620,
            ...
            2084203519, 2052019348, 2095170348, 2073864012, 2036520178,
            2072640359, 2057328940, 2079536840, 2082357614, 2051370426],
           dtype='int64', name='accnum', length=10000)
```

The last of this accounts would be the label `call_list[9999]` (2051370426). The number of subscriptions achieved is now: 

```
In [22]: df['cum_subscription'][call_list[9999]]
Out[22]: 4031
```
