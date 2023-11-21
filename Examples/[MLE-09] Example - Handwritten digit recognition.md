# [MLE-09] Example - Handwritten digit recognition

## Introduction

This example deals with the classification of grayscale images of handwritten digits (28 pixels by 28 pixels), into 10 classes (0 to 9). The data are the **MNIST data**, a classic in the machine learning community, which has been around for almost as long as the field itself and have been very intensively studied. 

The MNIST data set contains 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and Technology (NIST) in the 1980s. They have been extensively used for benchmarking. You can think of "solving" MNIST as the "Hello World" of deep learning. As you become a machine learning practitioner, MNIST comes up over and over again, in scientific papers, blog posts, and so on.

## The data set

The data from the 70,000 images come together in the file `digits.csv` (zipped). Every row stands for an image. The first column is a label identifying the digit (0-9), and the other 784 columns correspond to the image pixels (28 $\times$ 28 = 784). The column name `ixj` must be read as the gray intensity of the pixel in row $i$ and column $j$ (in the images). These intensities are integers from 0 = Black to 255 = White (8-bit grayscale).

## Questions

Q1. Pick the first digit image (row 1). The 784 entries on the right of the label, from `1x1` to `28x28`, are the pixels' gray intensities. Pack these numbers as a vector and reshape that vector as a matrix of 28 rows and 28 columns. With the `matplotlib.pyplot` function `imshow()`, plot the corresponding image. This function will be using default colors which do not help here, so you can turn everything to gray scale by inputting `pyplot.gray()`. Your plot will have then black background, with the number drawn in white. Guess how to reverse this, so the image looks like white paper with a number drawn in black ink.

Q2. Repeat the exercise with other images. You don't need `pyplot.gray()` anymore.

Q3. Split the data in a training set with 60,000 samples and a test set with 10,000 samples.

Q4. Train and test a **decision tree classifier**, using these data. Control the growth of the tree with the argument `max_leaf_nodes=128`.

Q5. Train and test a **random forest classifier**, with  `max_leaf_nodes=128` and `n_ estimators=10`. Is it better than the decision tree model?

Q6. Change the specification of your random forest model to see whether you can improve its performance.

## Importing the data

As in the preceding examples, we use the Pandas function `read_csv()` to import the data from a GitHub repository. Since the email messages don't have an identifier, we leave Pandas to create a `RangeIndex`. The source file is zipped, but `read_csv()` can manage this without a specific argument, based on the file extension `.zip`.

```
In [1]: import numpy as np, pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'digits.csv.zip')
```

We calculate the shape of the data frame, to check that we have the data for the 70,000 images.

```
In [2]: df.shape
Out[2]: (70000, 785)
```

## Target vector and feature matrix

We set this first column (the image labels) as the target vector. We can examine this vector with the Pandas function `value_counts()`. This shows that the data are a bit unbalanced: ones are most frequent, and fives least frequent.

```
In [3]: y = df.iloc[:, 0]
   ...: y.value_counts()
Out[3]: 
1    7877
7    7293
3    7141
2    6990
9    6958
0    6903
6    6876
8    6825
4    6824
5    6313
Name: label, dtype: int64
```

The 784 columns containing the pixel intensities will integrate the feature matrix. We convert them to a 2D NumPy array with `values`, so we can work better on the first questions. We check then that the pixels values are also as expected, with the NumPy function `unique()`.

```
In [4]: X = df.iloc[:, 1:].values
   ...: np.unique(X)
Out[4]: 
array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249, 250, 251, 252, 253, 254, 255], dtype=int64)
```

## Q1. Plotting the first image

Every row corresponds to the image of a digit. Let us visualize this by plotting the images with Matplotlib. In the first row, the 784 entries, from 1 $\times$ 1 to 28 $\times$ 28, are the pixels' gray intensities. To plot the image, we have to reshape it as a 2D array with 28 rows and 28 columns. This can be done with method `.reshape()`.

```
In [5]: pic = X[0, :].reshape(28,28)
```

The `matplotlib.pyplot` function `imshow()` converts this array to a picture and displays it:

```
In [6]: from matplotlib import pyplot as plt
   ...: plt.imshow(pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-09.1.png)

These are the default colors displayed by `imshow()`. To turn them into gray scale, one can use the argument `cmap='gray'`.

```
In [7]: plt.imshow(pic, cmap='gray');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-09.2.png)

The gray scale can be set as the default by entering `plt.gray()`. Now, reversing the scale, we can show the picture as it were a digit written with black pencil on a white paper surface:

```
In [8]: plt.gray()
   ...: plt.imshow(255 - pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-09.3.png)

This five is far from caligraphic, but still recognizable by a human eye.

## Q2. Plotting other images

The second image of the data set is a zero:

```
In [9]: pic = X[1, :].reshape(28,28)
   ...: plt.imshow(255 - pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-09.4.png)

And the third one a four:

```
In [10]: pic = X[2, :].reshape(28,28)
    ...: plt.imshow(255 - pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/mle-09.5.png)

## Q3. Train-test split

We split the data set, so we can validate the successive classification models that we will try. We keep 10,000 pictures for testing, which is common practice with the MNIST data.

```
In [11]: from sklearn.model_selection import train_test_split
    ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7)
```

## Q4. Decision tree classifier

We start with a decision tree classifier, to get a benchmark for the ensemble models. We use an estimator from the class `DecisionTreeClassifier()`, of the scikit-learn subpackage `tree`. Given the size of the data set, we set `max_leaf_nodes=128`, to control the growth of the tree.

```
In [12]: from sklearn.tree import DecisionTreeClassifier
    ...: treeclf = DecisionTreeClassifier(max_leaf_nodes=128)
    ...: treeclf.fit(X_train, y_train)
Out[12]: DecisionTreeClassifier(max_leaf_nodes=128)
```

We calculate the accuracy on both training and test data. The accuracy can be a good way to evaluate this model, since the data set is quite balanced and we don't have any preference for a particular digit.

```
In [13]: round(treeclf.score(X_train, y_train), 3), round(treeclf.score(X_test, y_test), 3)
Out[13]: (0.812, 0.8)
```

The accuracy is poor, in particular if we think on using the model to scan zipcodes (five digits). Even if the tree is big, we don't find evidence of overfitting.

## Q5. Random forest classifier

Maintaining the specification for the tree size, we try first a random forest classifier with 10 trees.

```
In [14]: from sklearn.ensemble import RandomForestClassifier
    ...: rfclf1 = RandomForestClassifier(max_leaf_nodes=128, n_estimators=10)
    ...: rfclf1.fit(X_train, y_train)
    ...: round(rfclf1.score(X_train, y_train), 3), round(rfclf1.score(X_test, y_test), 3)
Out[14]: (0.905, 0.892)
```
The improvement is quite clear. No overfitting so far. We will increase now the number of trees for monitoring the rogress.

## Q6. Change the specification 

Setting the number of trees to 50:

```
In [15]: rfclf2 = RandomForestClassifier(max_leaf_nodes=128, n_estimators=50)
    ...: rfclf2.fit(X_train, y_train)
    ...: round(rfclf2.score(X_train, y_train), 3), round(rfclf2.score(X_test, y_test), 3)
Out[15]: (0.92, 0.911)
```

Still improving. With 100 trees (the default):

```
In [16]: rfclf3 = RandomForestClassifier(max_leaf_nodes=128, n_estimators=100)
    ...: rfclf3.fit(X_train, y_train)
    ...: round(rfclf3.score(X_train, y_train), 3), round(rfclf3.score(X_test, y_test), 3)
Out[16]: (0.924, 0.917)
```

Slightly better, but the accuracy is getting flat. So, with a random forest classifier, we can easily achieve 90% accuracy. You can try variations on the size and the number of trees, doing a bit better. For instance, most practitioners prefer using the parameter `max_depth`, but it is usually less effective that the corresponding maximum leaf nodes (note that $2^7 = 128$).

```
In [17]: rfclf4 = RandomForestClassifier(max_depth=7, n_estimators=100)
    ...: rfclf4.fit(X_train, y_train)
    ...: round(rfclf4.score(X_train, y_train), 3), round(rfclf4.score(X_test, y_test), 3)
Out[17]: (0.914, 0.904)
```

So, using higher values of the parameter `max_leaf_nodes` looks more promising. Our final model is:

```
In [18]: rfclf5 = RandomForestClassifier(max_leaf_nodes=256, n_estimators=100)
    ...: rfclf5.fit(X_train, y_train)
    ...: round(rfclf5.score(X_train, y_train), 3), round(rfclf5.score(X_test, y_test), 3)
Out[18]: (0.945, 0.934)
```
