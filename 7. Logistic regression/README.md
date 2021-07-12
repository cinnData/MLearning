# 7. Logistic regression

### Logistic regression

Logistic regression is one of the simplest classification methods. Note that, in spite of its name, it is a classification method, not a regression method. The explanation is that logistic regression was created by statisticians, and regression does not mean the same in statistics and machine learning.

Let us start with **binary classification**, calling the target values positive and negative. A **logistic regression equation** is one of type

<img src="https://render.githubusercontent.com/render/math?math=\large p = F\big(b_0 %2B b_1 X_1 %2B b_2 X_2 %2B \cdots %2B b_k X_k\big).">

Here, *p* is the **predictive score**, that is, the probability of a positive target value, and *F* is the **logistic function**,

<img src="https://render.githubusercontent.com/render/math?math=\large F(x) = \displaystyle \frac{1}{1 %2B \exp(-x)}\,.">

![](https://github.com/cinnData/MLearning/blob/main/7.%20Logistic%20regression/fig%207.1.png)

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

`plt.savefig('Dropbox/ML Course/fig 7.1.png');`

As given by the logistic function, the scores fall within the unit interval (0 < *p* < 1). Although statisticians take them as probabilities, in machine learning you may be more pragmatic, using the scores just to rank the samples in order to select those to which a specific policy is going to be applied.

The coefficients of the logistic regression equation are obtained so that a certain loss function, called the **cross-entropy**, achieves its minimum value. In scikit-learn, you can choose the optimization method, named the **solver**, but this is a bit too mathematical for most users, so you better use the default, unless you are an optimization expert. If you are using Python, but you want logistic regression with a statistical flavor, you can use the package `StatsModels`.

### Logistic regression in scikit-learn

The scikit-learn subpckage `linear_model`, already mentioned in these notes, provides various regression and classification estimator. As usual in scikit-learn, we instantiate the estimator, which, in this case, we pick in the class `LogisticRegression`.  

`from sklearn.linear\under model import LogisticRegression`

`logclf = LogisticRegression()`

The basic methods `fit`, `predict`, `predict_proba` and `score` are available here, the same as in other classifiers. `score` returns the **accuracy**, that is the proportion of right prediction.

`LogisticRegression` works the same in a **multi-class** context, but instead of basing the predictions on a single equation, uses several logistic regression equations (as many equations as the number of target values minus 1).
