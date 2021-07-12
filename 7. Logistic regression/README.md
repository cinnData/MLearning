# 7. Logistic regression

### Logistic regression

Logistic regression is one of the simplest classification methods. Note that, in spite of its name, it is a classification method, not a regression method. The explanation is that logistic regression was created by statisticians, and regression does not mean the same in statistics and machine learning.

Let me start with binary classification, and suppose that the target $Y$ is coded as a dummy (1 = positive, 0 = negative) and that there is a collection of numeric features, $X_1,\ \dots,\ X_k$. A **logistic regression equation** is one of type

<img src="https://render.githubusercontent.com/render/math?math=\large p = F\big(b_0 %2B b_1 X_1 %2B b_2 X_2 %2B \cdots %2B b_k X_k\big).">

Here, $p$ is the predictive score, that is, a score for $Y=1$, and $F$ is the **logistic function**,

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




