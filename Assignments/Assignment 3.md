# Assignment 3

## Introduction

This assignment is a continuation of the analysis performed in the example **Term deposits**. Here, we take a different approach to the problem of **class imbalance**. With a 11.7% **conversion rate**, the data from the Portuguese bank showed a moderate class imbalance, which was addressed using a **scoring** approach. In this assignment, we use a **resampling** approach, training our predictive models in a modified data set in which the class imbalance has been artificially corrected.

## Questions

Q1. Perform a **random split** of the data set, taking one half for training and the other half for testing. You will resample the training subset, but leaving the testing subset as it is, without correcting there the class imbalance. 

Q2. Undersample the training subset, by randomly dropping as many negative units as needed to match the positive units, so that you end up with a pefectly balanced data set. Train a **logistic regression model** on these data and evaluate it on the testing subset. Supposing that you are going to call 20% of your clients, how would you validate your model?

Q3. Oversample the training subset, by randomly adding as many duplicates of the positive units as needed to match the negative units, so that you end up with a pefectly balanced data set. Train a logistic regression model on these data and evaluate it on the testing subset. Supposing that you are going to call 20% of your clients, how would you validate your model?
