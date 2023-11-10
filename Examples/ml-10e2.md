# [ML-10E2] Deep learning examples

## Introduction

This **deep learning** exercise uses the MNIST data that appeared in the example **Handwritten digit recognition**, in which we compared several ensemble classifiers. Assuming that the evaluation of the model is based on the accuracy, we show how to improve the predictive performance of neural network models by switching from the classic MLP neural network to a **convolutional neural network**. We use standard `tensorflow.keras` procedures to develop these models.

## Questions

Q1. Split the data in a training set with 60,000 samples and a test set with 10,000 samples.

Q2. Train and test a MLP model, with a hidden layer of 32 nodes, using these data.

Q3. How do you extract the predicted digits from this model?

Q4. Convert the gray scale (0 = Black, 255 = White) to (0 = Black, 1 = White), and try again with the same MLP model.

Q5. Try now with a convolutional neural network. Do ou get a real improvement with this *deep* model?

## Importing the data

We import to the MNIST data to a Pandas data frame, as we did previously.

```
In [1]: import numpy as np, pandas as pd
   ...: path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
   ...: df = pd.read_csv(path + 'digits.csv.zip')
```

## Target vector and feature matrix

We set the first column (image labels) as the target vector and the pixel intensities as the feature matrix. We use only NumPy arrays in this example, to simplify the syntax.

```
In [2]: y = df.iloc[:, 0].values
   ...: X = df.iloc[:, 1:].values
```

## Q1. Train-test split

```
In [3]: from sklearn import model_selection
   ...: X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1/7, random_state=0)
```

## Q2. MLP model

```
In [4]: from tensorflow.keras import models, layers
```

```
In [5]: net1 = [layers.Dense(32, activation='relu'), layers.Dense(10, activation='softmax')]
```

```
In [6]: clf1 = models.Sequential(net1)
```

```
In [7]: clf1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
```

```
In [8]: clf1.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));
Epoch 1/20
1875/1875 [==============================] - 1s 642us/step - loss: 1.7580 - acc: 0.6203 - val_loss: 0.9735 - val_acc: 0.7821
Epoch 2/20
1875/1875 [==============================] - 1s 557us/step - loss: 0.7216 - acc: 0.8159 - val_loss: 0.6070 - val_acc: 0.8551
Epoch 3/20
1875/1875 [==============================] - 1s 553us/step - loss: 0.4880 - acc: 0.8776 - val_loss: 0.4244 - val_acc: 0.8932
Epoch 4/20
1875/1875 [==============================] - 1s 538us/step - loss: 0.3731 - acc: 0.9086 - val_loss: 0.3681 - val_acc: 0.9094
Epoch 5/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.3244 - acc: 0.9179 - val_loss: 0.3757 - val_acc: 0.9136
Epoch 6/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.3020 - acc: 0.9231 - val_loss: 0.3205 - val_acc: 0.9235
Epoch 7/20
1875/1875 [==============================] - 1s 521us/step - loss: 0.2829 - acc: 0.9282 - val_loss: 0.3046 - val_acc: 0.9235
Epoch 8/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.2697 - acc: 0.9309 - val_loss: 0.3210 - val_acc: 0.9212
Epoch 9/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.2634 - acc: 0.9327 - val_loss: 0.3157 - val_acc: 0.9202
Epoch 10/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.2627 - acc: 0.9337 - val_loss: 0.3116 - val_acc: 0.9253
Epoch 11/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.2580 - acc: 0.9352 - val_loss: 0.3232 - val_acc: 0.9226
Epoch 12/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.2520 - acc: 0.9360 - val_loss: 0.3715 - val_acc: 0.9163
Epoch 13/20
1875/1875 [==============================] - 1s 536us/step - loss: 0.2492 - acc: 0.9363 - val_loss: 0.3294 - val_acc: 0.9227
Epoch 14/20
1875/1875 [==============================] - 1s 535us/step - loss: 0.2458 - acc: 0.9376 - val_loss: 0.3125 - val_acc: 0.9241
Epoch 15/20
1875/1875 [==============================] - 1s 535us/step - loss: 0.2400 - acc: 0.9394 - val_loss: 0.3095 - val_acc: 0.9271
Epoch 16/20
1875/1875 [==============================] - 1s 521us/step - loss: 0.2399 - acc: 0.9396 - val_loss: 0.3200 - val_acc: 0.9241
Epoch 17/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.2381 - acc: 0.9394 - val_loss: 0.3196 - val_acc: 0.9244
Epoch 18/20
1875/1875 [==============================] - 1s 542us/step - loss: 0.2387 - acc: 0.9398 - val_loss: 0.3207 - val_acc: 0.9245
Epoch 19/20
1875/1875 [==============================] - 1s 539us/step - loss: 0.2347 - acc: 0.9408 - val_loss: 0.3238 - val_acc: 0.9302
Epoch 20/20
1875/1875 [==============================] - 1s 531us/step - loss: 0.2325 - acc: 0.9416 - val_loss: 0.3716 - val_acc: 0.9240
```

```
In [9]: clf1.summary()
Model: "sequential"
-----------------------------------------------------------------
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 32)                25120     
                                                                 
 dense_1 (Dense)             (None, 10)                330       
                                                                 
=================================================================
Total params: 25450 (99.41 KB)
Trainable params: 25450 (99.41 KB)
Non-trainable params: 0 (0.00 Byte)
-----------------------------------------------------------------
```

## Q3. Prediction with a MLP network

```
In [10]: clf1.predict(X_test[:1, :])
1/1 [==============================] - 0s 74ms/step
Out[10]: 
array([[9.8956162e-01, 2.5109919e-08, 3.2807675e-03, 2.7105544e-04,
        2.8752672e-04, 2.0526997e-03, 3.6389669e-03, 3.6512902e-05,
        3.1564130e-05, 8.3928910e-04]], dtype=float32)
```

```
In [11]: y_test[0]
Out[11]: 0
```

## Q4. Rescaling the data

```
In [12]: X = X/255
```

```
In [13]: X_train, X_test = model_selection.train_test_split(X, test_size=1/7, random_state=0)
```

```
In [14]: clf2 = models.Sequential(net1)
    ...: clf2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
    ...: clf2.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test));
Epoch 1/20
1875/1875 [==============================] - 1s 574us/step - loss: 0.5411 - acc: 0.8475 - val_loss: 0.2654 - val_acc: 0.9226
Epoch 2/20
1875/1875 [==============================] - 1s 587us/step - loss: 0.2215 - acc: 0.9359 - val_loss: 0.2141 - val_acc: 0.9367
Epoch 3/20
1875/1875 [==============================] - 1s 561us/step - loss: 0.1882 - acc: 0.9453 - val_loss: 0.1982 - val_acc: 0.9390
Epoch 4/20
1875/1875 [==============================] - 1s 544us/step - loss: 0.1710 - acc: 0.9505 - val_loss: 0.1900 - val_acc: 0.9437
Epoch 5/20
1875/1875 [==============================] - 1s 539us/step - loss: 0.1595 - acc: 0.9535 - val_loss: 0.1854 - val_acc: 0.9470
Epoch 6/20
1875/1875 [==============================] - 1s 541us/step - loss: 0.1507 - acc: 0.9554 - val_loss: 0.1754 - val_acc: 0.9485
Epoch 7/20
1875/1875 [==============================] - 1s 538us/step - loss: 0.1451 - acc: 0.9570 - val_loss: 0.1822 - val_acc: 0.9462
Epoch 8/20
1875/1875 [==============================] - 1s 535us/step - loss: 0.1390 - acc: 0.9595 - val_loss: 0.1750 - val_acc: 0.9490
Epoch 9/20
1875/1875 [==============================] - 1s 532us/step - loss: 0.1346 - acc: 0.9603 - val_loss: 0.1799 - val_acc: 0.9484
Epoch 10/20
1875/1875 [==============================] - 1s 541us/step - loss: 0.1307 - acc: 0.9609 - val_loss: 0.1702 - val_acc: 0.9496
Epoch 11/20
1875/1875 [==============================] - 1s 529us/step - loss: 0.1270 - acc: 0.9621 - val_loss: 0.1767 - val_acc: 0.9508
Epoch 12/20
1875/1875 [==============================] - 1s 522us/step - loss: 0.1240 - acc: 0.9638 - val_loss: 0.1760 - val_acc: 0.9497
Epoch 13/20
1875/1875 [==============================] - 1s 547us/step - loss: 0.1208 - acc: 0.9643 - val_loss: 0.1707 - val_acc: 0.9502
Epoch 14/20
1875/1875 [==============================] - 1s 534us/step - loss: 0.1186 - acc: 0.9648 - val_loss: 0.1687 - val_acc: 0.9504
Epoch 15/20
1875/1875 [==============================] - 1s 528us/step - loss: 0.1157 - acc: 0.9655 - val_loss: 0.1736 - val_acc: 0.9504
Epoch 16/20
1875/1875 [==============================] - 1s 524us/step - loss: 0.1139 - acc: 0.9667 - val_loss: 0.1719 - val_acc: 0.9505
Epoch 17/20
1875/1875 [==============================] - 1s 536us/step - loss: 0.1130 - acc: 0.9658 - val_loss: 0.1684 - val_acc: 0.9520
Epoch 18/20
1875/1875 [==============================] - 1s 536us/step - loss: 0.1096 - acc: 0.9677 - val_loss: 0.1696 - val_acc: 0.9513
Epoch 19/20
1875/1875 [==============================] - 1s 538us/step - loss: 0.1081 - acc: 0.9678 - val_loss: 0.1691 - val_acc: 0.9521
Epoch 20/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.1061 - acc: 0.9683 - val_loss: 0.1726 - val_acc: 0.9501
```

## Q5. Convolutional neural network

```
In [15]: Z_train, Z_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
```

```
In [16]: net2 = [layers.Conv2D(32, (3, 3), activation='relu'),
    ...:     layers.MaxPooling2D((2, 2)),
    ...:     layers.Conv2D(64, (3, 3), activation='relu'), 
    ...:     layers.MaxPooling2D((2, 2)),
    ...:     layers.Conv2D(64, (3, 3), activation='relu'),
    ...:     layers.Flatten(),
    ...:     layers.Dense(64, activation='relu'),
    ...:     layers.Dense(10, activation='softmax')]
```

```
In [17]: clf3 = models.Sequential(net2)
    ...: clf3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
    ...: clf3.fit(Z_train, y_train, epochs=10, validation_data=(Z_test, y_test));
Epoch 1/10
1875/1875 [==============================] - 14s 7ms/step - loss: 0.1442 - acc: 0.9548 - val_loss: 0.0635 - val_acc: 0.9796
Epoch 2/10
1875/1875 [==============================] - 14s 8ms/step - loss: 0.0458 - acc: 0.9858 - val_loss: 0.0486 - val_acc: 0.9854
Epoch 3/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0327 - acc: 0.9899 - val_loss: 0.0463 - val_acc: 0.9848
Epoch 4/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0266 - acc: 0.9915 - val_loss: 0.0343 - val_acc: 0.9885
Epoch 5/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0197 - acc: 0.9935 - val_loss: 0.0306 - val_acc: 0.9910
Epoch 6/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0161 - acc: 0.9949 - val_loss: 0.0418 - val_acc: 0.9872
Epoch 7/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0126 - acc: 0.9958 - val_loss: 0.0398 - val_acc: 0.9893
Epoch 8/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0114 - acc: 0.9962 - val_loss: 0.0366 - val_acc: 0.9911
Epoch 9/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0104 - acc: 0.9966 - val_loss: 0.0300 - val_acc: 0.9926
Epoch 10/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0080 - acc: 0.9973 - val_loss: 0.0363 - val_acc: 0.9919
```

```

In [18]: clf3.summary()
Model: "sequential_2"
-----------------------------------------------------------------

 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2  (None, 13, 13, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 5, 5, 64)          0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 64)          36928     
                                                                 
 flatten (Flatten)           (None, 576)               0         
                                                                 
 dense_2 (Dense)             (None, 64)                36928     
                                                                 
 dense_3 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 93322 (364.54 KB)
Trainable params: 93322 (364.54 KB)
Non-trainable params: 0 (0.00 Byte)
-----------------------------------------------------------------
```
