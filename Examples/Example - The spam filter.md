# Example - The spam filter

## Introduction

A **spam filter** is an algorithm which classifies e-mail messages as either spam or non-spam, based on a collection of **numeric features** such as the frequency of certain words or characters. In a spam filter, the **false positive rate**, that is, the proportion of non-spam messages wrongly classified as spam, must be very low.

## The data set

The file `spam.csv` contains data on 4,601 e-mail messages. Among these messages, 1,813 have been classified as spam. The data were gathered at Hewlett-Packard by merging: (a) a collection of spam e-mail from the company postmaster and the individuals who had filed spam, and (b) a collection of non-spam e-mail, extracted from filed work and personal e-mail.

The variables are:

* 48 numeric features whose names start with `word_`, followed by a word. They indicate the frequency, in percentage scale, with which that word appears in the message. Example: for a particular message, a value 0.21 for `word_make` means that 0.21% of the words in the message match the word 'make'.

* 3 numeric features indicating, respectively, the average length of uninterrupted sequences of capital letters (`cap_ave`), the length of the longest uninterrupted sequence of capital letters (`cap_long`) and the total number of capital letters in the message (`cap_total`).

* A dummy indicating whether that e-mail message is spam (`spam`).

Source: Hewlett-Packard. Taken from T Hastie, R Tibshirani & JH Friedman (2001), *The Elements of Statistical Learning*, Springer.

## Questions

Q1. Develop a spam filter based on a **decision tree** classifier with **maximum depth** 3 and evaluate its performance. Repeat the exercise allowing maximum depths 4 and 5.

Q2. Which features are more relevant for filtering spam?
