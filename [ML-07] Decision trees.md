# [ML-06] Decision trees

## What is a decision tree?

A **decision tree** is a collection of **decision nodes**, connected by branches, extending downwards from the **root node**, until terminating in the **leaf nodes**. The usual graphical representation of a decision tree puts the root on top and the leaves at the bottom, as in Figures 1 and 2, which have been created with a scikit-learn utility and saved to PNG files with the `matplotlib.pyplot` function `savefig`.

Decision trees can be used for both classification and regression purposes. A decision tree creates a partition of the data set into a collection of subsets, one for each leaf. In a predictive model based on a decision tree, the predicted target value is the same for all the samples of the same leaf. More specifically, in a **decision tree regressor**, the predicted target value is the average target value in that leaf. In a **decision tree classifier**, the predicted probability class is the proportion of occurrence of that class in the leaf. Under the **default prediction rule**, the predicted class is the one that occurs more frequently in that leaf.

### Decision trees in scikit-learn

There are various ways to train a decision tree model from data. The top popular one is the **CART** (Classification And Regression Trees) algorithm. In scikit-learn, the subpackage `tree` provides the estimator classes `DecisionTreeRegressor` and `DecisionTreeClassifier`, both based on CART.

At every decision node, there is a **split**, based on one of the features and a cutoff value. CART chooses at every node the **optimal split**, that minimizes a **loss function**. In decision tree regressors, as in linear regression, the loss is the **mean square error** (MSE).

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_6.1.png)

Figure 1 shows a decision tree regressor, developed to predict the assessed value of a house (see the example *Assessing home values in West Roxbury*). At every node, you find the number of samples, the MSE and the predicted value, which is the mean target value in that leaf. The tree is optimal (meaning minimum MSE) among those satisfying the conditions set by the arguments of `DecisionTreeRegressor` (in this case `max_depth=2`).

In a decision tree classifier, the loss function is either the **Gini impurity measure** (the default) or the **entropy measure**. For every possible split, CART calculates the loss as the weighted average of the losses at the two branches, choosing the split that leads to the minimum loss.

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_6.2.png)

Figure 2 shows a decision tree classifier used as a spam filter (see the example *The spam filter*). At every leaf, you find the number of samples, the Gini value and the number of negative and positive samples (alphabetical order) in that leaf. In the leaf nodes, the predicted probabilities are the class proportions. In a binary setting, we can say that the predicted score for a sample is the proportion of positive samples in the leaf where that sample is. The tree is optimal in the sense that the total Gini value (the weighted average of the Gini values of the leaf nodes) is minimum.

### Controlling the growth of the tree

Predictive models based on decision trees are prone to **overfitting**. Even with a moderate number of features, a tree whose growth is not stopped can lead to a complex model with overfitting problems. In scikit-learn, the classes `DecisionTreeRegressor` and `DecisionTreeClassifier` allow many arguments for controlling the growth of the tree: `max_depth`, `max_leaf_nodes`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, etc. Only the first two will appear in these notes:

* The **depth** of a tree is the number of nodes in the longest branch. The trees of Figures 1 and 2 have been obtained by setting `max_depth=2`.

* `max_leaf_nodes` controls directly the **maximum number of leaves**.

To obtain the tree of Figure 1, we would use:

```
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(max_depth=2)
treereg.fit(X, y)
```

Then, we would plot the tree with:

```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(13,7))
plot_tree(treereg, fontsize=11);
```

To obtain the tree of Figure 2, we would use:

```
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=2)
```

### Feature importance

One of the advantages of decision tree algorithms is that it is very easy to get a report on **feature importance**. The importance of a feature is computed as the proportion of impurity decrease (either a mean sum of squares or a Gini value) brought by that feature. In scikit-learn, the attribute `feature_importances_` is a 1D array containing importance values for all the features. A zero value signals a feature that has not used in the tree.

```
treeclf.feature_importances_
```
