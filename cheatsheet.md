### Cheatsheet

#### Input/output

- `read.csv` and `write.csv`. Import/export data from/to CSV files. By default, string data are imported as factors, but that can be stopped with `stringsAsFactors=FALSE`.

#### Coercion

`as.numeric`, `as.character` and `as.factor`. Sometimes objects of different type are coerced in `R` to a common type. For instance, if we apply `sum` to a logical vector, the vector is coerced to numeric type previous to apply `sum`. This is implicit coercion. The user can also explicitly coerce a vector to a different type with an "as" function.

#### Exploring a data frame

- `head` and `tail`. Fist and last rows of a data frame. The default is 6 rows.

- `str`. Structure of an object. Applied to a data frame, it tells the type of data for each column.

- `dim`, `nrow` and `ncol`. Number of rows and/or columns of a data frame.

- `summary`. Output differs depending on the type object to which it is applied. Applied to a data frame gives summary stats for the non-character columns of the data frame.

- `names`. Some `R` objects are just lists where other are packed. The function `names` gives the names of the objects in the list. For instance, a data frame is a list of vectors of the same length. So, applied to a data set, `names` gives the names of the vectors that form the data frame.

#### Pivot tables

- `table`. Counts the number of occurrences for each of the possible values of one or two columns of a data frame. It does not include the missing values (`NA`).

- `tapply`. It has three arguments: A column of a data frame that we wish to summarize by groups, the grouping variable and the function to be applied in the summary (e.g. the mean).

#### Statistics

- `cor`. Applied to a data frame, it produces the correlation matrix of that data frame. Applied to a pair of vectors, it gives the correlation of the two vectors. Warning: It gives trouble if there are missing values (NA). Check `help(cor)` in that case.

- `mean`, `median`, `sd` and `sum`. They give the mean, median, standard deviation and sum, respectively.  They return `NA`  if there are missing values. With the argument `na.rm=TRUE`, missing values are ignored.

#### Plotting

- `barplot`. Bar plot for a numeric variable.

- `hist`. Histogram of a numeric variable. Arguments `main`, `xlab` and `ylab` specify the title and the labels for the X and Y axis, respectively.

- `plot`. It can take one or two variables. The default is `type="p"`, which uses dots. The size and shape of the dots can be changed with the argument `pch`. `type="l"` returns a line plot.

#### Regression equations

- `lm`. It returns an `lm` object, which is a list containing various objects related to linear regression, such as the coefficients, the predicted values, the residuals, etc. There many ways of specifying a linear model. Probably the simplest approach is to use the two arguments: (a) `formula`, to specify the equation, as in `formula = y ~ x1 + x2` and (b) `data`, as in `data = df`.

- `glm`. Overall function for fitting *generalized linear models* (GLM) to data. An additional argument allows the user to specify the particular model desired, within the GLM collection. With the argument `family="binomial"`, it returns a *logistic regression model*, which can be used for classification purposes.

- `predict`. In general, this function can be applied to a variety of predictive models, giving *predicted values*. In the simplest version, it uses two arguments: (a) `object`, to specify the name of the model, and (b) `newdata`, to specify the name of the data frame used for the predictions, which can be the same data frame on which the model was obtained, or a fresh one. In some applications, an additional argument is needeed, to specify the type of prediction desired. For instance, in a logistic regression model, the argument `type="response"` is used to obtain *predictive scores* in the 0--1 range.

#### Functions

- `function`. Defines a function. The syntax is `fname <- function(x) expression`. The expression involves the argument `x` and objects already defined. If the definition involves several steps, the expression is deployed in several lines and these lines are enclosed by curly braces (`{}`). In that case, it is better to include within the brackets a last line in which it is specified, with `return(value)`, what the function returns. Functions can have more than one argument, as in `f <- function(x, y) x*y/(x^2+y^2)`.

- `Vectorize`. A *vectorized* function is one that, when applied to a vector, works on every term of the vector separately and returns a vector of the same length. Many native functions of `R`, like `exp` or `sqrt` and operators like `+` or `*` are vectorized. User-defined functions can be vectorized with the function `Vectorize`.

#### Optmization

- `optimize`. Given a function of one numeric argument and an interval, it returns the minimum value and the argument on which the minimum value is achieved. With the additional argument `maximum=TRUE`, it returns the maximum instead of the mimimum.

#### Outer product of arrays

- `outer`. Given two vectors and a (vectorized) function of two arguments, it returns a matrix whose `[i, j]` entry is the value of the function on the i-th term of the first vector and the j-th term of the second vector.
