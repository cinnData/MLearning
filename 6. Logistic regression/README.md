# 6. Logistic regression

### Classification in scikit-learn

In classification, the terms of the target vector `y` are taken as labels for groups or classes. String data type is admitted for `y`. As in any supervised learning scikit-learn estimator, we have the three basic methods, `fit`, `predict` and `score`. But, in addition to the method `predict`, which returns a vector with labels to be matched to those of `y`, there is a variation here, the method `predict_proba`, which is relevant for many applications, specially in binary classification. 

`predict_proba` returns a 2d array, with as many columns as different target values are found in `y`, sorted alphabetically, and as many rows as `X`. The term in row *i* and column *j* of `y_proba` is read as the probability that the *i*-th sample belongs to the *j*-th target class. For every row, the sum of the probabilities equals 1. Different models differ in the way in which they calculate these probabilities. In the **logistic regression classifier**, which is the object of this chapter, they are the outcome of a (nonlinear) equation.

The *i*-th term of `y_pred`, as returned by `predict`, is the value with the highest probability. This makes sense as the default prediction, but, in some cases, specially when the distribution of the target values in uneven, you may want to change the way in which the predicted values are assigned. This point is developed below.

The proportion of samples classified in the right way, that is, those for which `y_pred` and `y` coincide, is called the **accuracy**. This is the value returned by `score` for a classification model. The accuracy may not be the best way to evaluate a model in real business applications, as discussed below.

### The confusion matrix

The evaluation of a classifier is usually based on a **confusion matrix**, obtained by crosstabulating the actual target values and the predicted target values. There is not a universal consensus on what to put in the rows and the columns. We use the same convention as the scikit-learn manual, which places the actual class in the rows and the predicted class in the columns. The same rule is followed in Géron (2017) and VanderPlas (2017). 

One way to get the confusion matrix is by means of the function `confusion_matrix`, from the subpackage `metrics`:

`from sklearn.metrics import confusion_matrix`

`conf = confusion_matrix(y, y_pred)`

The accuracy is the sum of the diagonal terms in this matrix divided by the sum of all terms.

### Binary classification

In **binary classification**, the two target values are typically called **positive** and **negative**. The labels positive/negative must be assigned so that they favor your intuition. Note that, if you leave this to the computer, it may call positive what you see as negative.

We take in this case the probability given by the classification model for the positive target value as a **predictive score**. In the default prediction, the samples whose scores exceed 0.5 are classified as positive and the rest as negative. But, in some cases, you may wish to replace 0.5 by a different **cutoff value**. In a business application, the choice of the cutoff may be based on a **cost/benefit analysis**. Specialized software can find the **optimal cutoff** for a user-specified cost matrix. 

Suppose that the target values positive/negative are coded as 1/0, and that we set the cutoff value as {\web cutoff}. Then the predictive scores are calculated as

For a binary classifier the four cells of the table are referred to as **true positive** (`y = 1`, `y_pred = 1`), **false positive** (`y = 0`, `y_pred = 1`), **true negative** (`y = 0`, `y_pred = 1`) and **false negative** (`y = 1`, `y_pred = 0`).

Although it looks as the obvious metric for the evaluation of a classifier, the accuracy is not always adequate, specially when the data present **class imbalance**. For instance, if you have 90% of negative samples in your data set, classifying all the samples as negative gives you 90% accuracy (you don't need machine learning for that!). Class imbalance is frequent in applications like credit scoring or fraud detection. It is also frequent in direct marketing, since conversion rates are typically low.

In a business context, a visual inspection of the confusion matrix is always recommended. It will probably help you to decide whether the classifier is going to be useful. In many cases, it is practical to examine the performance of the classifier separately on the actual positives and the actual negatives. Then, the usual metrics are:

* The **true positive rate**, or proportion of right classification among the actual positives,

	`tp = sum((y == 1) & (y_pred == 1))/sum(y == 1)`

* The **false positive rate**, or proportion of wrong classification among the actual negatives,

	`fp = sum((y == 0) & (y_pred == 1))/sum(y == 0)`

In a good model, the true positive rate should be high and the false positive rate low. Nevertheless, the relative importance given to these statistics depends on the actual application. The advantage of this approach is that it works when the proportion of positive samples in the training data has been artificially inflated. This may sound strange, but it is recommended for training classifiers which are intended to detect rare events, such as fraud. If this were the case, the training samples could not be taken as *representative* of the current population, and the accuracy derived from the confusion matrix would not be extrapolable to the real world. Nevertheless, the true positive and false negative rates would still be valid, since they are calculated separately on the actual positives and the actual negatives.

### Logistic regression

Logistic regression is one of the simplest classification methods. Note that, in spite of its name, it is a classification method, not a regression method. The explanation is that logistic regression was created by statisticians, and regression does not mean the same in statistics and machine learning.

Let us start with binary classification. A **logistic regression equation** is one of type

<img src="https://render.githubusercontent.com/render/math?math=\large p = F\big(b_0 %2B b_1 X_1 %2B b_2 X_2 %2B \cdots %2B b_k X_k\big).">

Here, *p* is the predictive score and *F* is the **logistic function**,

<img src="https://render.githubusercontent.com/render/math?math=\large F(x) = \displaystyle \frac{1}{1 %2B \exp(-x)}\,.">

![](https://github.com/cinnData/MLearning/blob/main/6.%20Logistic%20regression/fig%206.1.png)

The graph of the logistic function has an inverted S shape, as shown in Figure 1. You can get this figure as follows.

`import numpy as np`

`import matplotlib.pyplot as plt`

`from scipy.stats import logistic`

`x = np.linspace(-5, 5, 1000)`

`y = logistic.cdf(x)`

`plt.figure(figsize=(7,7))`

`plt.title('Figure 1. Logistic function', fontsize=16)`

`plt.plot(x, y, color='black', linewidth=1)`

`plt.tick_params(axis='both', which='major', labelsize=12)`

As given by the logistic function, the scores fall within the unit interval (0 < *p* < 1). Although statisticians take them as probabilities, in machine learning you may be more pragmatic, using the scores just to rank the samples in order to select those to which a specific policy is going to be applied.

The coefficients of the logistic regression equation are obtained so that a certain loss function, called the **cross-entropy**, achieves its minimum value. In scikit-learn, you can choose the optimization method, named the **solver**, but this is a bit too mathematical for most users, so you better use the default, unless you are an optimization expert. If you are using Python, but you want logistic regression with a statistical flavor, you can use the package `StatsModels`.

### Logistic regression in scikit-learn

The scikit-learn subpckage `linear_model` provides various regression and classification estimators. As usual in scikit-learn, we instantiate the estimator, which, in this case, we pick in the class `LogisticRegression`. 

`from sklearn.linear\under model import LogisticRegression`

`logclf = LogisticRegression()`

Next, we fit the estimator to the data:

`logclf.fit(X, y)`

If we accept the default prediction:

`y_pred = logclf.predict(X)`

But we may decide to use the predictive scores with an appropriate cutoff. Then

`scores = logclf.predict_proba(X)[:, 1]`

Note that the probability of a positive value is the second column of `logclf.predict_proba(X)`, because the columns are ordered alphabetically. The predicted target vector is calculated, now as

`y_pred = (scores > cutoff).astype('int')`

`LogisticRegression` works the same in a **multi-class** context, but instead of basing the predictions on a single equation, uses several logistic regression equations (as many equations as the number of target values minus 1).
