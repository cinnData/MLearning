# [MLE-11] Example - Airline passenger satisfaction

## Introduction

The data for this example, published by MysaR Ahmad Bhat, provide details of customers who have already flown with an **airline company**. The feedback of the customers on various context and their flight data has been consolidated.

The main objective of this data could be to predict whether a future customer would be satisfied with their service given the values of the other parameters. A second objective could to explore which aspects of the services offered have to be emphasized to generate more satisfied customers.

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

* `time`, satisfaction level with the convenience of the flight departure and arrival times from 1 (lowest) to 5 (highest). 

* `online_book`, satisfaction level with the online booking experience from 1 (lowest) to 5 (highest). 

* `checkin`, satisfaction level with the check-in service from 1 (lowest) to 5 (highest). 

* `online_board`, satisfaction level with the online boarding experience from 1 (lowest) to 5 (highest). 

* `gate`, satisfaction level with the gate location in the airport from 1 (lowest) to 5 (highest). 

* `on_board`, satisfaction level with the on-boarding service in the airport from 1 (lowest) to 5 (highest).

* `seat`, satisfaction level with the comfort of the airplane seat from 1 (lowest) to 5 (highest). 

* `leg_room`, satisfaction level with the leg room of the airplane seat from 1 (lowest) to 5 (highest). 

* `clean`, satisfaction level with the cleanliness of the airplane from 1 (lowest) to 5 (highest). 

* `food`, satisfaction level with the food and drinks on the airplane from 1 (lowest) to 5 (highest). 

* `in_flight`, satisfaction level with the in-flight service from 1 (lowest) to 5 (highest). 

* `wifi`, satisfaction level with the in-flight Wifi service from 1 (lowest) to 5 (highest). 

* `entertain`, satisfaction level with the in-flight entertainment from 1 (lowest) to 5 (highest). 

* `baggage`, satisfaction level with the baggage handling from the airline from 1 (lowest) to 5 (highest). 

* `sat`, overall satisfaction level with the airline (Satisfied=1, Neutral or unsatisfied=0).

Source of the data: Kaggle. Rows with missing data have been deleted. The option 'Not applicable' was available in the questionnaire for the satisfaction levels measured in 1-5 range, but it was encoded as 3.

## Questions

Q1. Develop a **random forest** model for predicting the passenger satisfaction.

Q2. The same for a **XGBoost** model. Is the improvement relevant?

Q3. Which features are most relevant for predicting the passenger satisfaction?

Q4. Finally, try a **multilayer perceptron** model. Is there an improvement? 

Q5. Does the MLP model get better after **normalizing** the features?

## Importing the data

As in other examples, we use the Pandas funcion `read_csv()` to import the data from a GitHub repository. Since the passengers don't have an identifier, we leave Pandas to create a `RangeIndex`. 

```
In [1]: import pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'airsat.csv')
```

## Exploring the data

The data report printed by the method `.info()` does not contradict the description given above.

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

As in lother examples of supervised learning, we create a target vector and a feature matrix. The target vector is the last column (`sat`) and the feature matrix is made with the other columns.

```
In [4]: y = df['sat']
   ...: X = df.drop(columns='sat')
```

## Train-test split

The data set is randomly split, keeping a 20% of the data units for testing. The argument `random_state=0` will ensure the reproducibility of the split.

```
In [5]: from sklearn.model_selection import train_test_split
   ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Q1. Random forest model

Our first predictive model is a random forest model. Using the scikit-learn class `ensemble.RandomForestClassifier()`, we set the number of tress to 200 and the maximum depth to 5. The model is trained on the training data.

```
In [6]: from sklearn.ensemble import RandomForestClassifier
   ...: rf = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
   ...: rf.fit(X_train, y_train)
Out[6]: RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
```

We evaluate the model on both the training and the test subsets, since this comes easily in scikit-learn.

```
In [7]: rf.score(X_train, y_train).round(3), rf.score(X_test, y_test).round(3)
Out[7]: (0.913, 0.911)
```

The accuracy is about 91%. No evidence of overfitting.

## Q2. XGBoost model

We repeat the exercise with a XGBoost model, using the class `XGBClassifier()` from the package `xgboost`. We use the same arguments as in the preceding section, leaving the learning rate at the default value. 

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

Overfitting is moderate. We set the reference value for the accuracy at 95%. This is, more or less, what you can achieve with a shallow model, taking the features as they come, with no engineering of any type.

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

Online boarding looks that the most relevant feature. Also, the model reveals a difference between flying for business and the overall satisfaction, although it does not show in which direction. Popular wisdom tells us that people flying for personal issues (and also paying) are usually less tolerant with anything not working properly. Indeed, this is what cross tabulation tells us:

```
In [11]: pd.crosstab(df['business'], df['sat'])
Out[11]: 
sat           0      1
business              
0         27387   3129
1         33150  49819
```

## Q4. MLP model

We try now a simple neural network, using the TensorFlow with the Keras API. We import the modules `models` and `layers`, that contain all the resources needed.

```
In [12]: from tensorflow.keras import models, layers
```

Next, we specify the **network architecture**, as a list. This will be multilayer perceptron (MLP) with one hidden layer. The **input layer** contains one node for every feature, and will be specified on the fly when we fit the model to the data. The **hidden layer** is then the first item of the list. It has 32 nodes (powers of 2 are commonly used in deep learning), and the activation function is the **rectified linear unit function** (ReLU). The **output layer** has two nodes, since this model is a binary classifier. The activation function is the **softmax function**, which ensures that the output is a vector of probabilities (positive numbers summing 1). These two layers are **dense layers**, meaning that every node is connected to all nodes of the preceding layer. This means 22 $\times$ 32 $\times$ 2 connections.

```
In [13]: network = [layers.Dense(32, activation='relu'), layers.Dense(2, activation='softmax')]
```

The next step is instantiate an object of the class `models.Sequential()`. This works as in scikit-learn. The list `network` is the value of the parameter `layers`.

```
In [14]: mlp = models.Sequential(layers=network)
```

Now, we **compile** the model, meaning the mathematical apparatus neede for fitting the model to the data gets ready. We have to specify the **optimization algorithm** (`optimizer='adam'`), the **loss function** (`loss='sparse_categorical_crossentropy'`) and the metrics used to evaluate the model performance (`metrics='acc'`).

```
In [15]: mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
```

We are ready now to apply the method `.fit()`. Note that the number of iterations is specified here, not when creating the model `mlp`, as in scikit-learn. In every iteration, or **epoch**, the data set is randomly split in batches of size 32 (if you accept the deafult size). These batches are passed sequentially, and the weights are updated for evey batch. This means that they not updated 50 times, but 50 times the number of batches (2,838). With `verbose=0` we stop a report of the progress being gradually displayed on the screen (we will see this later). The semicolon stops an irrelevant output showing up (the same we do when plotting with Matplotlib)). In this example, takes every epoch takes between one and two seconds in a regular laptop.

```
In [16]: mlp.fit(X_train, y_train, epochs=50, verbose=0);
```

Once the model has been trained, it is evaluated on the test data. The model does not bring an improvement on the accuracy, compoared to the ensemble models tried before.  

```
In [17]: round(mlp.evaluate(X_test, y_test, verbose=0)[1], 3)
Out[17]: 0.895
```

## Q5. Multilayer perceptron model (normalized data)

scikit-learn has a method for normalizing all the columns of the feature matrix in one shot. But it is not dificult to do it in Pandas. First we define a min-max normalization function:

```
In [18]: def normalize(x): 
    ...:     return (x - x.min())/(x.max() - x.min())
```

Now, we apply it by column with the method `.apply()`.

```
In [19]: XN = X.apply(normalize)
```

We have now a new feature matrix, that we split exactly in the same way as we did with `X` and `y` (the argument `random_state=0` does the trick).

```
In [20]: XN_train, XN_test = train_test_split(XN, test_size=0.2, random_state=0)
```
We replicate the process of question Q4 with the normalized features. The improvement is quite clear, though the MLP falls a bit of the XGBoost model.

```
In [21]: mlp = models.Sequential(network)
     ...: mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
     ...: mlp.fit(XN_train, y_train, epochs=50, verbose=0);
     ...: round(mlp.evaluate(XN_test, y_test, verbose=0)[1], 3)
Out[21]: 0.94
```
