# 9. Model validation

### Overfitting

**Overfitting** is a typical problem of predictive models. It occurs when the model fits satisfactorily the data used to obtain it, but fails with data that have not been used. The purpose of **validation** is to dismiss the concerns about overfitting raised by the use of complex machine learning models. These concerns are well justified, since many popular models, like neural nets, are prone to overfit the training data. Validation is also called **out-of-sample testing**, because this is what it really does.

In the simplest approach to validation, we obtain the model from a **training data set**, trying it on a **test data set**. The training and test sets can have a temporal basis (eg training with the first ten months of the year and testing with the last two months), or they can be obtained from a random split of a unique data set.

**Cross-validation** is more sophisticated. Though it has many variations, this course only covers *k*-**fold cross-validation**, in which the original data set is randomly partitioned into *k* equally sized subsets. Of the *k* subsets, one is used for testing and the other *k* - 1 subsets for training. The model is scored on the test data (eg with the accuracy). This process is repeated for each of the *k* subsets. The *k* evaluation scores obtained can then be averaged to produce a single value, which is taken as the score of the model. *k* = 10 has been recommended by some authors, but *k* = 10 is also popular.

In scikit-learn, the subpackage `model_selection` provides tools for validating supervised learning models. I start by the simplest approach, based on the function `train_test_split`. In a second round, I'll show you how to perform *k*-fold cross-validation with the function `cross_val_score`.

### Train-test split

Suppose that you are given a target vector `y` and a feature matrix `X`. A **train-test split** can be obtained with:

`from sklearn.model_selection import train_test_split`

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`


Setting `test_size=0.2`, you get a 80-20 split, which is quite popular among practitioners. Of course, there is nothing special in a 20% test size. The idea is that it is not reasonable to waste too much data on testing.

For instance, suppose, that `logclf` is a `LogisticRegression` instance, with the default arguments. The model coefficients are obtained by applying `logclf.fit` to the training data:

`logclf.fit(X_train, y_train)`

Then, you can evaluate the model on both data sets and compare the results. Suppose that your evaluation is based on the true positive and false positive rates, and the cutoff has been set to 0.5, so you get your predictions directly from `logclf.predict`. You can extract a true positive and a false positive rate from the training set:

`ypred_train = logclf.predict(X_train)`

`from sklearn.metrics import confusion_matrix`

`conf_train = confusion_matrix(y_train, ypred_train)`

`tp_train = conf_train[1, 1]/sum(conf_train[1, :])`

`fp_train = conf_train[0, 1]/sum(conf_train[0, :])`

In a similar way, in the test set:

`ypred_test = logclf.predict(X_test)`

`conf_test = confusion_matrix(y_test, ypred_test)`

`tp_test = conf_test[1, 1]/sum(conf_test[1, :])`

`fp_test = conf_test[0, 1]/sum(conf_test[0, :])`

Roughly speaking, the model can be considered as validated if `tp_test` and `fp_test` are as good as `tp_train` and `fp_train`. You may find this a bit loose, but it is the approach we frequently take in business.

If your evaluation is based on the accuracy, you will just compare the two numbers `logclf.score(X_train, y_train)` and `logclf.score(X_test, y_test)`.

### Cross-validation

As far as you can trust the evaluation of your model to the method `score` (either *R*-squared or accuracy), *k*-fold cross-validation provides a quick and dirty assesment of your model. You can do it with:

`from sklearn.model_selection import cross_val_score`

`cross_val_score(logclf, X, y, cv=10)`

This function returns a vector of ten scores (accuracy values, in this case). You can average them to get a score for the model, but you can also take a look at the variation across folds, to decide whether you will trust the model. The argument `cv=10` sets the number of folds. `. 

*Note*. In some cases, people complain about cross-validation in scikit-learn producing unreasonable results, such as negative score values. This is due to the way it has been programmed, so many of us prefer to try a series of train-test splits.
