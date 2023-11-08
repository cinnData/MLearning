# [ML-10E] Example - Airline passenger satisfaction

## Introduction

The data for this example, published by Sayantan Jana, provide details of customers who have already flown with an **airline company**. The feedback of the customers on various context and their flight data has been consolidated.

The main purpose of this data set is to predict whether a future customer would be satisfied with their service given the details of the other parameters values. A second objective is to explore which aspects of the services offered have to be emphasized to generate more satisfied customers.

## The data set

The file `airsat.csv` contains data on 113,485 customers. The columns are:

* `female`, gender of the passenger (Female=1, Male=0).

* `age`, age of the passenger. Only passengers older than 15 were included in the data collection.

* `first`, type of airline customer (First-time=1, Returning=0).

* `business`, purpose of the flight (Business=1, Personal=0).

* `busclass`, travel class for the passenger seat (Business=1, Economy=0).

* `distance`, flight distance in miles.

* `depdelay`, flight departure delay in minutes.

* `arrdelay`, flight arrival delay in minutes.

* `time`, satisfaction level with the convenience of the flight departure and arrival times from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `online_book`, satisfaction level with the online booking experience from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `checkin`, satisfaction level with the check-in service from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been  encoded as 3.

* `online_board`, satisfaction level with the online boarding experience from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `gate`, satisfaction level with the gate location in the airport from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `on_board`, satisfaction level with the on-boarding service in the airport from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `seat`, satisfaction level with the comfort of the airplane seat from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `leg_room`, satisfaction level with the leg room of the airplane seat from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `clean`, satisfaction level with the cleanliness of the airplane from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `food`, satisfaction level with the food and drinks on the airplane from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `in_flight`, satisfaction level with the in-flight service from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `wifi`, satisfaction level with the in-flight Wifi service from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `entertain`, satisfaction level with the in-flight entertainment from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `baggage`, satisfaction level with the baggage handling from the airline from 1 (lowest) to 5 (highest). The option 'Not applicable' was available, but has been encoded as 3.

* `sat`, overall satisfaction level with the airline (Satisfied=1, Neutral or unsatisfied=0).

## Questions

Q1. Develop a randon forest model for predicting the passenger satisfaction.

Q2. Try next a XGBoost model. Is the improvement relevant?

Q3. Which features are most relevant for prdicting the passenger satisfaction?

Q4. Finally, try a MLP model. Is there an improvement? 

## Importing the data

As in other examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. Since the passengers don't have an identifier, we leave Pandas to create a `RangeIndex`. 

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'airsat.csv')
```

## Exploring the data

The data report printed by the method `.info()` does not contradict the description presented above.

```
In [2]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 113485 entries, 0 to 113484
Data columns (total 23 columns):
 #   Column        Non-Null Count   Dtype
---  ------        --------------   -----
 0   female        113485 non-null  int64
 1   age           113485 non-null  int64
 2   first         113485 non-null  int64
 3   business      113485 non-null  int64
 4   busclass      113485 non-null  int64
 5   distance      113485 non-null  int64
 6   depdelay      113485 non-null  int64
 7   arrdelay      113485 non-null  int64
 8   time          113485 non-null  int64
 9   online_book   113485 non-null  int64
 10  checkin       113485 non-null  int64
 11  online_board  113485 non-null  int64
 12  gate          113485 non-null  int64
 13  on_board      113485 non-null  int64
 14  seat          113485 non-null  int64
 15  leg_room      113485 non-null  int64
 16  clean         113485 non-null  int64
 17  food          113485 non-null  int64
 18  in_flight     113485 non-null  int64
 19  wifi          113485 non-null  int64
 20  entertain     113485 non-null  int64
 21  baggage       113485 non-null  int64
 22  sat           113485 non-null  int64
dtypes: int64(23)
memory usage: 19.9 MB
```

The proportion of satisfied passengers is quite close to 50%, so class imbalance is not an issue here. We will use the accuracy to evaluate the models obtained.

```
In [3]: df['sat'].mean().round(3)
Out[3]: 0.467
```

## Target vector and feature matrix

We create a target vector and a feature matrix. The target vector is the last column (`sat`) and the feature matrix is made with the other columns.

```
In [4]: y = df['sat']
   ...: X = df.drop(columns='sat')
```

## Train-test split

```
In [5]: from sklearn.model_selection import train_test_split
   ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Q1. Random forest model

```
In [6]: from sklearn.ensemble import RandomForestClassifier
   ...: rf = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
   ...: rf.fit(X_train, y_train)
Out[6]: RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
```

```
In [7]: rf.score(X_train, y_train).round(3), rf.score(X_test, y_test).round(3)
Out[7]: (0.913, 0.911)
```

## Q2. XGBoost model

```
In [8]: from xgboost import XGBClassifier
   ...: xgb = XGBClassifier(max_depth=5, n_estimators=200, random_state=0)
   ...: xgb.fit(X_train, y_train)
Out[8]: 
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=5, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=200, n_jobs=None,
              num_parallel_tree=None, random_state=0, ...)
```

```
In [9]: xgb.score(X_train, y_train).round(3), xgb.score(X_test, y_test).round(3)
Out[9]: (0.969, 0.953)
```

## Q3. Relevant features

In any predictive model based on decision trees, the relevance of the different features for predicting the target can be assessed with the attribute `.feature_importances_`, which works in `xgboost` as in scikit-learn. Since it is a plain 1D array, without index labels, we convert it to a Pandas series, using the column names as the index. Sorting by values, we get a clear report.

```
In [10]: pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
Out[10]: 
online_board    0.380214
business        0.149885
wifi            0.082065
busclass        0.078103
first           0.074857
entertain       0.034292
checkin         0.031343
seat            0.023297
clean           0.020647
baggage         0.020112
gate            0.018233
leg_room        0.017903
in_flight       0.015224
on_board        0.013118
time            0.008011
age             0.007870
online_book     0.007446
arrdelay        0.005196
food            0.003832
distance        0.003288
depdelay        0.002653
female          0.002411
dtype: float32
```

## Q4. MLP model

```
In [11]: from tensorflow.keras import models, layers
```

```
In [12]: network = [layers.Dense(32, activation='relu'), layers.Dense(2, activation='softmax')]
```

```
In [13]: mlp = models.Sequential(network)
```

```
In [14]: mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
```


```
In [15]: mlp.fit(X_train, y_train, epochs=50, verbose=0);
```

```
In [16]: round(mlp.evaluate(X_test, y_test, verbose=0)[1], 3)
Out[16]: 0.895
```

## Q5. Multilayer perceptron model (normalized data)

```
In [17]: def normalize(x): 
    ...:     return (x - x.min())/(x.max() - x.min())
```

```
In [18]: XN = X.apply(normalize)
```

```
In [19]: XN_train, XN_test = train_test_split(XN, test_size=0.2, random_state=0)
```

```
In [20]: mlp = models.Sequential(network)
     ...: mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
     ...: mlp.fit(XN_train, y_train, epochs=50, verbose=0);
     ...: round(mlp.evaluate(XN_test, y_test, verbose=0)[1], 3)
Out[21]: 0.94
```