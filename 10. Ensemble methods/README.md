# 10. Ensemble methods

### Ensemble methods

Suppose that you ask a complex question to thousands of random people, and then aggregate their answers. In many cases, you will find that this aggregated answer is better than an expert's answer. This has been called the **wisdom of the crowds**.

**Ensemble learning** is based on a similar idea. If you aggregate the predictions of a group of regression or classification models, you will often get better predictions than with the best individual model:

* Suppose that you have trained a few regression models, each one achieving a moderate correlation. A simple way to get better predictions could be to average the predictions of these models.

* In a classification context, you would average the class probabilities. This is called **soft voting**, in contrast to **hard-voting**, which consist in picking the class getting more votes. This course only covers soft voting.

The group of models whose predictions are aggregated is called an **ensemble**. In scikit-learn, the subpackage `ensemble` offers plenty of choice. On top of the popularity ranking, we find the random forest and the gradient boosting methods, both using ensembles of decision tree models.

### Random forests

One way to get a diverse ensemble is to use the same algorithm for every predictor, but training it on different random subsets of the training set. When these subsets are extracted by sampling with replacement, the method is called **bagging** (short for bootstrap aggregating). Bagging allows training instances to be sampled several times for the same predictor.

The star of bagging ensemble methods is the **random forest** method, which allows extra randomness when growing trees, by using just a random subset of the features at every split. This results in a greater tree diversity, generally yielding an overall better model. Despite its simplicity, random forest models are among the most powerful predictive models available. In general, they have less overfitting problems than other models.

Random forests can be used for both regression and classification. In the scikit-learn subpackage `ensemble`, this is provided by the estimator classes `RandomForestRegressor` and `RandomForestClassifier`. The growth of the trees is controlled with arguments such as `max_depth` and `max_leaf_nodes` (no defaults), as in individual tree models. You can also control the number of trees with the argument `n_estimators` (the default is 100 trees), and the number of features that can be used at every split with the argument `max_features` (look at the manual if you wish to play with this argument).

Here follows a classification example:

`from sklearn.ensemble import RandomForestClassifier`

`rfclf = RandomForestClassifier(max_leaf_nodes=128, n_estimators=50)`

In general, increasing the tree size or the number of trees leads to better prediction, but this may come at the price of overfitting the training data. A safe approach is to accept the default of `n_estimators=100`, increasing gradually the tree size and testing overfitting at every step.

### Gradient boosting

The general idea of the **boosting methods** is to train the models of the ensemble sequentially, each trying to correct its predecessor. Among the many boosting methods available, the star is the **gradient boosting** method, which can be used in both regression and classification. As in the random forest method, the models of the ensemble are based on decision trees, though, here, every tree model is fit to the errors made by the previous one.  

The prediction of the ensemble model is obtained as a weighted average. The weights decrease at every step according to a parameter called the **learning rate**. With a low learning rate, the weight decreases more slowly. There is a trade-off between the learning rate and the number if trees. With a low learning rate, you will probably need a higher number of trees. Some experts recommend to set a low learning rate (in the range from 0.001 to 0.01) and aim at a high number of trees (in the range from 3,000 to 10,000), but for that you may need huge computing power, since gradient boosting is a slow process.

In scikit-learn, gradient boosting is provided by `GradientBoostingRegressor` and `GradientBoostingClassifier`, from the subpackage `ensemble`. The growth of the trees and the number of trees is controlled as in the random forest. Most practitioners accept the defaults `n_estimators=100` and `learning_rate=0.1`, but go beyond the default `max_depth=3`.

Here follows a classification example:

`from sklearn.ensemble import GradientBoostingClassifier`

`gbclf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)`

**XGBoost** (extreme gradient boosting) is is an implementation of gradient boosting designed for speed and performance. It has recently been on top of the ranking in applied machine learning competitions. For Python, it is available in the package `xgboost`, which can be used as if it were a scikit-learn subpackage (though other interfaces are available). Gradient boosting optimization takes less time in `xgboost` than in the scikit-learn subpackage. The defaults are `n_estimators=100`, `learning_rate=0.3` and `max_depth=6`.

The `xgboost` version of the preceding example would be:

`from xgboost import XGBClassifier`

`xgbclf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)`

`xgboost` can be installed from the shell, with `pip install xgboost` or `conda install -c conda-forge xgboost`. The second one works only for Anaconda distributions. Although it is much slower, it is recommended, because `xgboost` has many dependencies, and you may have a conflict of versions.
