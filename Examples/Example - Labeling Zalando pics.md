# Example - Labeling Zalando pics

## Introduction

The file `zalando.csv` contains data of 70,000 labeled 28  28 grayscale images of Zalando's articles. The labels are associated to the following articles: T-shirt/top (0), trousers (1), pullover (2), dress (3), coat (4), sandal (5), shirt (6), sneaker (7), bag (8) and ankle boot (9).

These data were intended to serve as a direct drop-in replacement for the MNIST data, for benchmarking machine learning algorithms. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating lightness or darkness. This pixel-value is an integer between 0 and 255 (0 = Black, 255 = White). In the data set, every row stands for an image. The label is in the first column, and the rest of the columns contain the pixel-values of the image.

## The data set

In the data set, every row stands for an image. The label is in the first column, and the rest of the columns contain the pixel-values of the image. These columns are named `pixel1`, `pixel2`, etc, the pixels being counted from top-left to bottom-right.

## Questions

Q1. Plot one image of every type with the `matplotlib.pyplot` function `imshow`, to guess how difficult classifying these images may be.

Q2. Split the data in a training set with 60,000 samples and a test set with 10,000 samples.

Q3. Train and test a random forest classifier, with `max_leaf_nodes=128` and `n_estimators=100`.

Q4. Train and test a neural network classifier with one hidden layer of 32 nodes. Compare your restults with those of the random forest model.

Q5. Add a second hidden layer and compare.
