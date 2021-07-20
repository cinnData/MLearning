# 8. One-hot encoding

### Dummy variables

Statisticians refer to the binary variables that take 1/0 values as **dummy variables**, or dummies. Dummy variables are used to enter **categorical variables**, whose values are labels for groups or categories, in regression equations.

In machine learning, **one-hot encoding** is a popular name for the process of extracting dummies from a set of a categorical features. Since dummies are frequently involved in data analysis, statistical software applications provide simple ways for extracting them. scikit-learn has a method for extracting dummies in a massive way, from several categorical features at a time. 

In applications like Excel, where you explicitly create new columns for the dummies, the rule is that the number of dummies is equal to the number of groups minus one. You set one of the groups as the **baseline group** and create a dummy for every other group. More specifically, suppose that *G*0 is the baseline group and *G*1, *G*2, etc, are the other groups. First, you define a dummy *D*1 for *G*1 as *D*1 = 1 on the samples from group *G*1, and *D*1 = 0 on the samples from the other groups. Then, you repeat this for *G*2, and the rest of the groups except the baseline, which leads to a set of dummies *D*1, *D*2, etc, which you can include in your regression equation.

Using Python, you do not have to care about all this. You just pack into a matrix  the categorical features that you wish to encode and apply to that matrix the appropriate method, which will return a matrix whose columns are the dummies. Note that categorical feature can have a numeric data type (eg the zipcode) or string type (eg gender, coded as F/M).

### One-hot encoding in scikit-learn

A one-hot encoding **transformer** is available in scikit-learn, in the class `OneHotEncoder` of the subpackage `preprocessing`. Suppose that we separate the categorical features in a matrix `X2`, and the rest in another matrix `X1`. The matrix `X2` will then be transformed as follows. 

First, we instantiate a `OneHotEncoder` transformer as usual in scikit-learn:

`from sklearn.preprocessing import OneHotEncoder`

`enc = OneHotEncoder()`

Now, we fit the transformer `enc` to the matrix `X2`:

`enc.fit(X2)`

We get the encoded matrix as:

`X2 = enc.transform(X2).toarray()`

Finally, we put the two feature matrices together with the NumPy function `concatenate`:

`X = np.concatenate([X1, X2], axis=1)`

Note that `axis=1` indicates the two matrices are concatenated horizontally. For this to be possible, they must have the same number of rows. The default of is `axis=0`, that is, vertical concatenation. 

