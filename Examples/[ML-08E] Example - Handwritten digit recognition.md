# [ML-08E] Example - Handwritten digit recognition

## Introduction

This example deals with the classification of grayscale images of handwritten digits (28 pixels by 28 pixels), into 10 classes (0 to 9). The data come from the **MNIST data set**, a classic in the machine learning community, which has been around for almost as long as the field itself and has been very intensively studied. 

The MNIST data set contains 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and Technology (NIST) in the 1980s. You can think of “solving” MNIST as the "Hello World" of deep learning, what you do to verify that your algorithms are working as expected. As you become a machine learning practitioner, you will see MNIST come up over and over again, in scientific papers, blog posts, and so on.

## The data set

The data from the 70,000 images come together in the file `digits.csv`. Every row stands for an image. The first column is a label identifying the digit (0-9), and the other 784 columns correspond to the image pixels (28 $\times$ 28 = 784). The column name `ixj` must be read as the gray intensity of the pixel in row $i$ and column $j$ (in the images). These intensities are integers from 0 = Black to 255 = White (8-bit grayscale).

## Questions

Q1. Pick the first digit image (row 1). The 784 entries on the right of the label, from `1x1` to `28x28`, are the pixels' gray intensities. Pack these numbers as a vector and reshape that vector as a matrix of 28 rows and 28 columns. With the `matplotlib.pyplot` function `imshow`, plot the corresponding image. `pyplot.imshow` will be using default colors which do not help here, so you can turn everything to gray scale by inputting `pyplot.gray()`. So, your plot will have black background and the number is drawn in white. Guess how to reverse this, so that the image looks like white paper with a number drawn in black ink.

Q2. Repeat the exercise with other images. You don't need `pyplot.gray()` anymore.

Q3. Split the data in a training set with 60,000 samples and a test set with 10,000 samples.

Q4. Train and test a decision tree classifier, with `max_leaf_nodes=128`, using these data.

Q5. Train and test a random forest classifier, with  `max_leaf_nodes=128` and `n_ estimators=10`. Is it better than the decision tree model?

Q6. Change the specification of your random forest model to see whether you can improve its performance.

## Importing the data

```
In [1]: import numpy as np, pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'digits.csv.zip')
```

```
In [2]: df.shape
Out[2]: (70000, 785)
```

## Target vector and feature matrix

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

```
In [5]: pic = X[0, :].reshape(28,28)
```

```
In [6]: from matplotlib import pyplot as plt
   ...: plt.imshow(pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e.1.png)

```
In [7]: plt.imshow(pic, cmap='gray');
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e.2.png)

```
In [8]: plt.gray()
   ...: plt.imshow(255 - pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e.3.png)

## Q2. Plotting other images

```
In [9]: pic = X[1, :].reshape(28,28)
   ...: plt.imshow(255 - pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e.4.png)


```
In [10]: pic = X[2, :].reshape(28,28)
    ...: plt.imshow(255 - pic);
```

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_8e.5.png)
