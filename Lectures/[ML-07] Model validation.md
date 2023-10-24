# [ML-07] Model validation

## Overfitting

**Overfitting** is a typical problem of predictive models. It occurs when a model fits satisfactorily the data used to obtain it, but fails with data which have not been used. The purpose of **validation** is to dismiss the concerns about overfitting raised by the use of complex machine learning models. These concerns are well justified, since many popular models, like neural nets, are prone to overfit the training data. Validation is also called **out-of-sample testing**, because this is what we really do.

In the simplest approach to validation, we derive the model from a **training data set**, trying it on a **test data set**. The training and test sets can have a temporal basis (*e.g*. training with the first ten months of the year and testing with the last two months), or they can be obtained by means of a random split of a unique data set.

For the top powerful prediction models, such as gradient boosting or deep learning models, overfitting is part of the process, so practitionerss take the metrics resulting from evaluating the model on the test data set as the valid ones. When this is applied systematically, the principle of testing with a data set which has not been used to obtain the model is violated. The standard approach to this problem is to use a third data set, the **validation data set**, in the model selection process, keeping the test set apart, to be used in a final evaluation.

**Cross-validation** is a more sophisticated approach. It has many variations, among them **$k$-fold cross-validation**, in which the original data set is randomly partitioned into $k$ equally sized subsets. One of the $k$ subsets is used for testing and the other  subsets for training, and the model is scored on the test data. This process is repeated for each of the  subsets. The resulting evaluation scores (either R-squared or accuracy) can then be averaged to produce a single value, which is taken as the score of the model. $k=10$ has been recommended by some authors, but $k=3$ and $k=5$ are more popular nowadays. If you mean to keep a test set apart from the process, you will split first the data in two, performing the cross-validation in one subset, while keeping the other subset for the final test. This approach is quite popular among practitioners.

## Train-test split in scikit-learn

In scikit-learn, the subpackage `model_selection` provides tools for validating supervised learning models. Suppose that you are given a target vector `y` and a feature matrix `X`. A random **train-test split** can be obtained with:

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Setting `test_size=0.2`, you get a 80-20 split, which is quite popular. Of course, there is nothing special in a 20% test size. The idea behind this partition is that it is not reasonable to waste too much data on testing. Also, note that you have to split twice if you wish to have a training set, a validation set and a test set.

For instance, suppose, that `clf` is a `LogisticRegression` instance. The coefficients of the equarion are obtained by applying `clf.fit` to the training data:

```
clf.fit(X_train, y_train)
```

Then, you can evaluate the model on both data sets and compare the results. Suppose that your evaluation is based on the accuracy. You train the model on the training set and evaluate it on both sets:

```
ypred_train = clf.predict(X_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
```

Overfitting happens when there is a relevant difference between these two metrics. If you have a collection of potential models, you will replace here the test set by the validation set, leaving the test set apart. By applying this process repeatedly, you can selecting the model with the best performance on the validation set, and then use the test set for the final evaluation.

## Cross-validation in scikit-learn

The subpackage `model_selection` also provides a toolkit for cross-validation. I restrict myself to the the simplest approach. Instead of fitting the classifier `clf` to any data, you use:

```
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=3)
```

The function `cross_val_score` returns a vector of three scores (accuracy values, in this case). While you can average these scores to get an overall score for the model, you may also take a look at the variation across folds, to decide whether you will trust the model. The argument `cv=3` sets the number of folds. The default is `cv=5`. You can use other metrics than the accuracy, for instance with the argument `scoring='precision'`.
