# [MLA-04] Assignment 4

## Introduction

This assignment is a continuation of the analysis performed in the example MLE-09, which uses the **MNIST data**. The example discusses the extent to which simple machine learning models can recognize handwritten **digits**.

## Questions

Q1. At every node of every tree, the **random forest** algorithm searches for the best split using a **random subset of features**. The number of features is controlled by the parameter `max_features`. We have used the default, which is the square root of the number of columns of the feature matrix (`max_features='sqrt`). This means, in this case, 28 features. Logic tells us that, by increasing `max_features`, we will improve the accuracy, but the learning process (the fit step) will get slower. Try some variations on this, to see how it works in practice. Do you think that using the default number of features here was a good choice?

Q2. Develop a **gradient boosting classifier** for these data, extracted from the `xgboost` class `XGBClassifier()`. Take into account that, with hundreds of columns, a gradient boosting model may be much slower to train than a random forest model with the same tree size and number of trees. A model with 100 trees and a size similar to those shown in this example can take one hour to train (less with XGBoost), though you may find a speed-up by increasing the **learning rate**.

Q3. Create a single feature matrix for the whole MNIST data set and extract ten clusters from it, using the **$k$-means** method. Which is the digit that is better matched by one of the clusters? Can you assign a digit to every cluster?
