# [ML-06] Logistic regression

## Class probabilities

In classification models, the predicted target values are usually obtained in two steps:

* For every sample, the model calculates a set of **class probabilities**, one for each class.

* The **predicted class** is the one with higher probability.

This is the **default prediction** method. When this approach is used, the class probabilities may be hidden, so the model is presented as it were extracting the predictions directly. The different types of models differ in the way in which they calculate these probabilities.

In some cases, we switch from the default prediction rule to one which uses the class probabilities in a different way. Departure from the default is not rare when the data present **class imbalance**, that is, when the proportion of samples of one class is significantly different from the proportion of samples of other classes. Class imbalance is frequent in applications like **credit scoring** or **fraud detection**. It is also frequent in **direct marketing**, since conversion rates are typically low.

## Binary classification

In **binary classification**, the two target values are typically called **positive** and **negative**. The labels positive/negative must be assigned so that they favor your intuition. Mind that, if you leave this to the computer, it may call positive what you regard as negative.

In a binary setting, using two complementary probabilities is not practical, so we focus on the positive class probability. This probability, called the **predictive score**, can be used for management purposes in many business applications (*e.g*. in credit scoring).

In the default prediction, a sample is classified as positive when its score exceeds 0.5. But you may wish to replace 0.5 by a different **threshold** value. In a business application, the choice of the threshold may be based on a **cost/benefit analysis**. It is not hard to find (approximately) the **optimal threshold** for a user-specified cost matrix.

## The confusion matrix
The evaluation of a classifier is usually based on a **confusion matrix**, obtained by cross tabulating the actual target values and the predicted target values. There is not a universal consensus on what to place in the rows and what in the columns. I use the same convention as the scikit-learn manual, with the actual target value in the rows and the predicted target value in the columns.

In a binary setting, a visual inspection of the confusion matrix is always recommended. It will probably help you to decide whether the model is going to be useful. In many cases, it is practical to examine the model performance separately on the actual positives and negative samples.

The four cells of the confusion matrix are referred to as **true positive** (actual positives predicted as positives), **false positive** (actual negatives predicted as positives), **true negative** (actual negatives predicted as negatives) and **false negative** (actual positives predicted as negatives).

| | Predicted positive | Predicted negative |
| --- | :---: | :---: |
| **Actual negative** | TN | FP |
| **Actual positive** | FN | TP |

The proportion of samples classified in the right way, that is, those for which the actual and the predicted values coincide, is called the accuracy,

$$\textrm{Accuracy} = \frac{\textrm{TN}+\textrm{TP}} {\textrm{TN}+\textrm{FP}+\textrm{FN}+\textrm{TP}}.$$

The accuracy can be calculated directly, or extracted from the confusion matrix, as the sum of the diagonal terms divided by the sum of all terms. Although it looks as the obvious metric for the evaluation of a classifier, the accuracy is not always adequate, specially when the data present class imbalance. For instance, if you have 90% of negative samples in your training data set, classifying all the samples as negative gives you 90% accuracy (you don't need machine learning for that!).

In a business context, a visual inspection of the confusion matrix is always recommended. It will probably help you to decide whether the classifier is going to be useful. In many cases, it is practical to examine the performance of the classifier separately on the actual positives and the actual negatives. Then, the usual metrics are:

* The **true positive** rate is the proportion of right classification among the actual positives,
$$\textrm{TP\ rate} = \frac{\textrm{TP}} {\textrm{TP}+\textrm{FN}}.$$

* The **false positive rate** is the proportion of wrong classification among the actual negatives,
$$\textrm{FP\ rate} = \frac{\textrm{FP}} {\textrm{FP}+\textrm{TN}}.$$

In a good model, the true positive rate should be high and the false positive rate low. The relative importance given to these statistics depends on the actual application. Their advantage is that they are still valid when the proportion of positive samples in the training data has been artificially inflated, because they are calculated separately on the actual positives and the actual negatives. This may sound strange, but it is recommended in cases of class imbalance. When the proportion of positive samples is inflated, the training data cannot be taken as representative of the current population, and the accuracy derived from the confusion matrix cannot be extrapolated to the real world.

An alternative to the true positive and false negative rates, used by scikit-learn, is based on the precision and the recall:

* The **precision** is the proportion of right classification among the predicted positives,
$$\textrm{Precision} = \frac{\textrm{TP}} {\textrm{TP}+\textrm{FP}}.$$

* The **recall** is the same as the true positive rate,
$$\textrm{Recall} = \frac{\textrm{TP}} {\textrm{TP}+\textrm{FN}}.$$

In a good model, precision and recall should be high. Some authors combine precision and recall in a single metric (in mathematical terms, it is the harmonic mean), called the **F1-score**, also available in scikit-learn:
$$\textrm{F1-score} = \frac{\textrm{2}\times\textrm{Precision}\times\textrm{Recall}} {\textrm{Precision}+\textrm{Recall}}
= \frac{2\,\textrm{TP}} {2\,\textrm{TP}+2\,\textrm{FP}+\textrm{FN}}.$$

## Logistic regression

**Logistic regression** is one of the simplest classfication methods. The class probabilities are calculated by means of a (nonlinear) equation. Note that, in spite of its name, it is a classfication method, not a regression method. The explanation is that logistic regression was created by statisticians, and regression does not mean the same in statistics and machine learning.

Let us start with binary classification. A logistic regression equation is one of type

$$p = F\big(b_0 + b_1X_1 + b_2X_2 +_ \cdots + b_kX_k\big).$$

Here, $p$ is the predictive score and $F$ is the **logistic function**, whose mathematical expression is

$$F(x) = \frac {1} {1 + \exp(-x)}.$$

The graph of the logistic function has an inverted S shape, as shown in the figure. As given by this function, the scores fall within the unit interval ($0 < p < 1$). Although statisticians take them as probabilities, in machine learning you may be more pragmatic, using the scores just to rank the samples in order to select those to which a specific policy is going to be applied.

![](figure/fig_6.1.png)

The coef cients of the logistic regression equation are optimal, meaning that a certain loss function, called the **cross-entropy**, extracted from information theory, achieves its minimum value. In scikit-learn, you can choose the optimization method, named the **solver**, but this is a bit too mathematical for most users. So, it is recommended to use the default, unless you are an optimization expert. If you are using Python, but you want logistic regression with a statistical package `StatsModels`.

## Classification in scikit-learn

In scikit-learn, we just find a few differences between classification and regression models. In classification, the target values (the terms of `y`) are taken as **labels** for groups or classes. Data type `str` is admitted for `y`. We also have here the three basic methods, `fit`, `predict` and `score`.

For instance, for a binary logistic regression model, we use the class `LogisticRegression` from the subpackage `linear_model`. As usual in scikit-learn, we instantiate an estimator:

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
```

Next, we fit the data:

```
clf.fit(X, y)
```

For a classifier, the method `score` gives you the accuracy of the default prediction:

```
acc = clf.score(X, y)
```

The default prediction is given by the method `predict`:

```
y_pred = clf.predict(X)
```

But, here, in addition to `predict`, you also have `predict_proba`, which returns a 2D array with the predicted class probabilities. For every row, the sum of the class probabilities equals 1.

If you use the predictive scores with a threshold `t`, the predicted target values will be obtained as:

```
y_score = clf.predict_proba(X)[:, 1]
y_pred = (y_score > t).astype(int)
```

Note that the positive class probabilities are in the second column of `clf.predict_proba(X)`, because the classes are ordered alphabetically. Now, the accuracy can be calculated directly, as:

```
sum(y == y_pred)
```

Denoting by `1` the positive class and by `0` the negative class, the true positive and the false positive rates will be:

```
tp_rate = sum((y == 1) & (y_pred == 1))/sum(y == 1)
fp_rate = sum((y == 0) & (y_pred == 1))/sum(y == 0)
```

Though you can calculate them directly, as suggested above, these and other metrics can also be extracted from the confusion matrix. One way to get it is by means of the function `confusion_matrix`, from the subpackage `metrics`:

```
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y, y_pred)
```

This subpackage also provides specific functions for the precision (`precision_score`), the recall (`recall_score`) and the F1-score (`f1_score`).

`LogisticRegression` works the same in a **multi-class** setting, but instead of basing the predictions on a single equation, it uses several logistic regression equations (as many equations as the number of target values minus 1).
