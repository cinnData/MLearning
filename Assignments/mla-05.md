# [MLA-05] Assignment 5

### Introduction

This exercise uses 70,000 labeled 28 $\times$ 28 grayscale images of Zalando's articles. The labels are associated to the following articles: T-shirt/top (0), trousers (1), pullover (2), dress (3), coat (4), sandal (5), shirt (6), sneaker (7), bag (8) and ankle boot (9).

These data were intended to serve as a direct drop-in replacement for the MNIST data, for benchmarking ML algorithms. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating lightness or darkness. This pixel-value is an integer between 0 and 255 (0 = Black, 255 = White). In the data set, every row stands for an image. The label is in the first column, and the rest of the columns contain the pixel-values of the image.

## The data set

The data from the 70,000 images come together in the file zalando.csv (zipped). Every row stands for an image. Every row stands for an image. The first column is a label identifying the digit (0-9), and the other 784 columns correspond to the image pixels (28 $\times$ 28 = 784). The column name `pixelnum` must be read as the gray intensity of the pixel `num`, counting pixels by row, from top-left to bottom-right. The intensities are integers from 0 = Black to 255 = White (8-bit grayscale).

## Questions

Q1. Print an image of each type, in gray scale, with white background, to see how difficult can be, for the human eye, to classify these images. 

Q2. Split the data in a training set with 60,000 images and a test set with 10,000 images. 

Q3. Develop an ensemble classifier for these images.

Q4. Develop a neural network classifier for these images and compare it to the ensemble classifier of question Q3. 

Q5. Which type of clothes are better predicted by these models?
