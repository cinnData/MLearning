# Example - Handwritten digit recognition

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
