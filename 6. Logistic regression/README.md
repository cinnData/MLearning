# 6. Logistic regression

### Class probabilities

In **classification models**, the predicted target values are usually obtained in two steps:

* For every sample, the model calculates a set of **class probabilities**, one for each class.

* The predicted class is the one with higher probability.

This is the default prediction method, and the class probabilities are frequently hidden, so the model can be presented as it were extracting the predictions directly. The different types of models differ in the way in which they calculate these probabilities. 

In some cases, we are interested in changing the default prediction rule by one which uses the class probabilities in a different way. This happens frequently in a binary classification context. Departure from the default prediction rule is not rare when the data present **class imbalance**, that is, when the proportion of samples of one class is significantly different from the proportion of samples of another class. Class imbalance is frequent in applications like credit scoring or fraud detection. It is also frequent in direct marketing, since conversion rates are typically low. 

### Binary classification

In **binary classification**, the two target values are typically called **positive** and **negative**. The labels positive/negative must be assigned so that they favor your intuition. Note that, if you leave this to the computer, it may call positive what you see as negative.

In a binary setting, using two complementary probabilities is not practical, so we focus on the positive class probability. This probability, called the **predictive score**, can be used for management purposes in many business applications (eg in credit scoring). 

In the default prediction, a sample is classified as positive when its score exceeds 0.5. But, in some cases, you may wish to replace 0.5 by a different **cutoff value**. In a business application, the choice of the cutoff may be based on a **cost/benefit analysis**. It is not hard to find (approximately) the **optimal cutoff** for a user-specified cost matrix.

### The confusion matrix

The evaluation of a classifier is usually based on a **confusion matrix**, obtained by crosstabulating the actual target values and the predicted target values. There is not a universal consensus on what to place in the rows and what in the columns. This course uses the same convention as the scikit-learn manual, with the actual target value in the rows and the predicted target value in the columns. 

The proportion of samples classified in the right way, that is, those for which the actual and the predicted values coincide, is called the **accuracy**. It can be calculated directly, or extracted from the confusion matrix, as the sum of the diagonal terms divided by the sum of all terms. Though the accuracy is used in most books and courses, it may not be the best way to evaluate a classification model in real business applications, as you will see in the examples of this course.

In a binary context, a visual inspection of the confusion matrix is always recommended. It will probably help you to decide whether the model is going to be useful. In many cases, it is practical to examine the model performance separately on the actual positives and the actual negatives.

The four cells of the confusion matrix are referred to as **true positive** (actual positives predicted as positives), **false positive** (actual negatives predicted as positives), **true negative** (actual negatives predicted as negatives) and **false negative** ((actual positives predicted as negatives). Then:

* The **true positive rate** is the proportion of right classification among the actual positives (TP/(TP + FN).

* The **false positive rate** is the proportion of wrong classification among the actual negatives (FP/(FP + TN)).

In a good model, the true positive rate should be high and the false positive rate low. The relative importance given to these statistics depends on the actual application. Their advantage is they are still valid when the proportion of positive samples in the training data has been artificially inflated, because they are calculated separately on the actual positives and the actual negatives. This may sound strange, but it is recommended in cases of class imbalance. If this were the case, the training data could not be taken as *representative* of the current population, and the accuracy derived from the confusion matrix would not be extrapolable to the real world.

### Logistic regression

**Logistic regression** is one of the simplest classification methods. The class proabbilities are calculated by means of a (nonlinear) equation.Note that, in spite of its name, it is a classification method, not a regression method. The explanation is that logistic regression was created by statisticians, and regression does not mean the same in statistics and machine learning.

Let us start with binary classification. A **logistic regression equation** is one of type:

<img src="https://render.githubusercontent.com/render/math?math=\large p = F\big(b_0 %2B b_1 X_1 %2B b_2 X_2 %2B \cdots %2B b_k X_k\big).">

Here, *p* is the predictive score and *F* is the **logistic function**, whose mathematical formula is:

<img src="https://render.githubusercontent.com/render/math?math=\large F(x) = \displaystyle \frac{1}{1 %2B \exp(-x)}\,.">

The graph of the logistic function has an inverted S shape, as shown in Figure 1. You can get this figure as follows.

`import numpy as np, matplotlib.pyplot as plt`

`from scipy.stats import logistic`

`x = np.linspace(-5, 5, 1000)`

`y = logistic.cdf(x)`

`plt.figure(figsize=(7,7))`

`plt.title('Figure 1. Logistic function', fontsize=16)`

`plt.plot(x, y, color='black', linewidth=1)`

`plt.tick_params(axis='both', which='major', labelsize=12)`

![](https://github.com/cinnData/MLearning/blob/main/6.%20Logistic%20regression/fig%206.1.png)

As given by the logistic function, the scores fall within the unit interval (0 < *p* < 1). Although statisticians take them as probabilities, in machine learning you may be more pragmatic, using the scores just to rank the samples in order to select those to which a specific policy is going to be applied.

The coefficients of the logistic regression equation are optimal, meaning that a certain loss function, called the **cross-entropy**, achieves its minimum value. In scikit-learn, you can choose the optimization method, named the **solver**, but this is a bit too mathematical for most users. So, you better use the default, unless you are an optimization expert. If you are using Python, but you want logistic regression with a statistical flavor, you can use the package `StatsModels`.

### Classification in scikit-learn

In scikit-learn, we find a just a few differences between classification and regression models. In classification, the terms of the target vector `y` are taken as **labels** for groups or classes. String data type is admitted for `y`. We also have the three basic methods, `fit`, `predict` and `score`. 

For instance, for a binary logistic regression model, we use the class `LogisticRegression` from the subpckage `linear_model`. As usual in scikit-learn, we instantiate an estimator:

`from sklearn.linear\under model import LogisticRegression`

`logclf = LogisticRegression()`

Next, we fit the data:

`logclf.fit(X, y)`

If we accept the default prediction:

`ypred = logclf.predict(X)`

But, here, in addition to `predict`, we have `predict_proba`, which returns a 2d array with class probabilities. For every row, the sum of the class probabilities equals 1. 

Another difference is that, for a classification model, the method `score` returns the accuracy. But this is only true if we use the default prediction method. If we use the predictive scores with an appropriate cutoff, the predicted target values will be obtained as:

`scores = logclf.predict_proba(X)[:, 1]`

`ypred = (scores > cutoff).astype('int')`

Note that class probabilities for positive value is the second column of `logclf.predict_proba(X)`, because the columns are ordered alphabetically. Now the accuracy can be calculated directly, as:

`np.sum(y == ypred)`

Denoting by 1 the positive class and by 0 the negative class, the true positive and the false positive rates will be:

`tp = np.sum((y == 1) & (ypred == 1))/sum(y == 1)`

`fp = np.sum((y == 0) & (ypred == 1))/sum(y == 0)`

These statistics can also be extracted from the confusion matrix. One way to get it is by means of the function `confusion_matrix`, from the subpackage `metrics`:

`from sklearn.metrics import confusion_matrix`

`conf = confusion_matrix(y, ypred)`

`LogisticRegression` works the same in a **multi-class** context, but instead of basing the predictions on a single equation, it uses several logistic regression equations (as many equations as the number of target values minus 1).
