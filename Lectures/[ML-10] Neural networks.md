#  [ML-10] Neural networks

## What is a neural network?

An (artificial) **neural network** is an interconnected set of computational elements, called **nodes** or neurons, organized in layers. Every connection of a node to another node has a **weight**. In machine learning, these weights are learned from the training data.

There are many types of neural networks. We start with the **multilayer perceptron** (MLP), which was the standard approach for many years, until **deep learning** rose to prominence. At that time, practitioners referred to the multilayer perceptron network when using the expressions neural network or artificial neural network (ANN).

Under the hood, a MLP model is just a set of equations, as explained below. MLP models can be used for both regression and classification. In scikit-learn, they can be extracted from the classes `MLPRegressor()` and `MLPClassifier()` of the subpackage `neural_networks`. In this lecture, we skip this, presenting directly the Keras API of the Python package TensorFlow.

## Basics of the MLP model

A multilayer perceptron is formed by:

* The **input layer**, whose nodes are the features used for the prediction.

* The **output layer**. In regression models, it has a unique node, which is the target (as in the figure below), while, in classification models, it has one node for every target value.

* A sequence of **hidden layers**, placed between the input and the output layers. If the network is **fully-connected**, that is, if every node of a layer is connected to all the nodes of the following layer, the model is completely specified by the number of hidden layers and the number of nodes in each hidden layer.

How do these networks work? Suppose first that $Z$ is a hidden node and $U_1, U_2, \dots, U_k$ are the nodes of the preceding layer. Then, the values of $Z$ are calculated as

$$Z = F\big(w_0 + w_1U_1 + w_2U_2 + \cdots + w_kU_k\big).$$

In this context, the slope coefficients $w_1, w_2, \dots, w_k$  are called weights, and the intercept $w_0$ is called **bias**. $F$ is the **activation function**. The role of the activation function is to introduce nonlinearity in the model (see below).

The multilayer perceptron could be seen as if the samples were circulating through the network one-by-one. The feature values are entered in the input nodes, which send them to the nodes of the first hidden layer. At each hidden node, they are combined using the corresponding weights, and the result is transformed by means of the activation function. The hidden nodes send the resulting values to the nodes of the next layer, where they are combined. According to the legend, this is the way animal neurons learn.

Let us help intuition with the graphical representation of a small network. The model of the figure below is a MLP regressor with one hidden layer of two nodes. The diagram is just a graphical representation of a set of three equations, two for the hidden nodes and one for the output node. The equation of node $A$ combines $X_1$, $X_2$ and $X_3$ with weights $w_{1A}$, $w_{2A}$ and $w_{3A}$, while the equation in node $B$ combines them with weights $w_{1B}$, $w_{2B}$ and $w_{3B}$. The biases are $w_{0A}$ and $w_{0B}$, respectively.

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig_10.1.png)

At the hidden nodes, the **activation function** is applied to the values given by these equations. Once the activation has been applied, the outcomes of the two hidden nodes are combined in the third equation, with weights $w_{AY}$ and $w_{BY}$ and bias $w_{0Y}$, to obtain the predicted value of $Y$. This model has a total of 11 parameters.

## The activation function

The choice of the activation function is based on performance, since we do not have any serious theory which could explain why a specific mathematical formula works better than others. Just a few years ago, the **logistic function** was the recommended activation function in the hidden layers, although some preferred a similar formula called the **hyperbolic tangent** function. The current trend favors the **rectified linear unit function** ($\hbox{ReLU}$). $\hbox{ReLU}(x)$ is equal to $x$ when $x>0$ and equal to $0$ otherwise. So, the default activation in the hidden layers consists in turning the negative incoming values into zeros.

In a MLP regressor (as in the figure), there is no activation at the (single) output node, so the equation predicting the values at that node is linear. In a MLP classifier, there are as many output nodes as target values. An activation function called the **softmax function** is applied to the whole set of incoming values, turning them into a set of **class probabilities**. The mathematical expressions involved in the definition of the softmax function are similar to the logistic function formula.

## Other technicalities

* *How to find the optimal weights*. Initially, the weights are randomly assigned. Then, an iterative process starts. At every step, the prediction is performed with the current weights, the value of a **loss function** is calculated, and the weights are adjusted in order to reduce the loss. The process is expected to converge to an optimal solution, but, in practice, a maximum number of iterations is pre-specified. In regression, the loss is usually the MSE, while, in classification, it is the **cross-entropy**. The adjustment of the weights starts at the last layer, and continues backwards until the input layer. This is called **backpropagation**.

* *The optimization method*, called **solver** in scikit-learn and **optimizer** in the Keras API. The current trend favors the **stochastic gradient descent** (SGD) method, which has many variants. Though you may find in books or tutorials the variant `optimizer='rmsprop'`, we use here `optimizer='adam'`, which is faster.

* *The number of iterations*, that is, the number of times every sample passes through the network is controlled in Keras with the parameter **epochs**. The default is `epochs=1`. The samples don't pass all at once, but in random batches (see below).

* *The learning rate*, which we have already found in gradient boosting modeling, is a parameter which rules how fast the adjustment of the weights is done. If it is too low, there is no convergence to the optimal solution. If it is too high, you can overshoot the optimal solution. Modern ML software allows setting an initial learning rate and decrease it as the learning process goes on. The Keras default is `learning_rate=0.001`. We don't use this parameter in this course

* *The batch size*. In the SGD method, the training data are randomly partitioned in batches in every iteration. The batches are tried one-by-one and the weights are modified every time that a batch is tried. The Keras default is `batch_size=32`. We don't use this parameter in this course, but if you do, you may speed up the training step by increasing the batch size.

* *Normalization*. Optimization methods are sensitive to feature scaling, so it is highly recommended to scale your data. In the old data mining suites, normalization was applied as a part of the algorithm, and the output was scaled back to the original range. It is not so in the Python ML toolbox.

## What is deep learning?

**Deep learning**, the current star of machine learning, is based on neural networks. The success of deep learning, not yet fully understood, is attributed to the ability of creating improved **representations** of the input data by means of successive layers of features.

Under this perspective, deep learning is a successful approach to **feature engineering**. Why is this needed? Because, in many cases, the available features do not provide an adequate representation of the data, so replacing the original features by a new set may be useful. At the price of oversimplifying a complex question, the following two examples may help to understand this:

* A **pricing model** for predicting the sale price of a house from features like the square footage of the plot and the house, the location, the number of bedrooms, the existence of a garage, etc. You will probably agree that these are, indeed, the features that determine the price, so they provide a good representation of the data, and a **shallow learning** model, such as a random forest regressor, would be a good approach. No feature engineering is needed here.

* A model for **image classification**. Here, the available features are related to a grid of pixels. But we do not classify an image based on specific pixel positions. Recognition is based on shapes and corners. A shape is a created by a collection of pixels, each of them close to the preceding one. And a corner is created by tho shapes intersecting in a specific way. So, we have the input layer of pixels, a first hidden layer of shapes providing a better representation, a second layer of corners providing an even better representation.

The number of hidden layers in a neural network is called the **depth**. But, although deep learning is based on neural networks with more than one hidden layer, there is more in deep learning than additional layers. In the MLP model as we have seen it, every hidden node is connected to all the nodes of the preceding layer and all the nodes of the following layer. In the deep learning context, these fully-connected layers are called **dense**. But there are other types of layers, and the most glamorous applications of deep learning are based on networks which are not fully-connected.

## TensorFlow and Keras

The mathematics involved in the specification of the neural networks used in deep learning are written in terms of **tensors**. A tensor is the same as a NumPy array. So, a 0D tensor is a scalar (a number), a 1D tensor is a vector, a 2D tensor is a matrix, etc. The operations with tensors and their properties are the object of **tensor algebra**, which is an extension of the matrix algebra which you may have found in undergraduate courses.

You do not need to worry about the math technicalities, as far as you pay attention to the specification of the **shape** of the input and output tensors in every layer of the network. Among the many attempts to implement the mathematics of the deep learning networks, including the tensors, the top popular one is **TensorFlow**, developed at Google Brain and released in 2015. The main competitor is **PyTorch**, released by Facebook. Recent news are that PyTorch is becoming the top choice in research (academic journals and conferences), although TensorFlow is still the undisputed leader in industry (according to job listings).

**Keras** is a deep learning framework for Python (there is also a version for R), which provides a convenient way to define and train deep learning models. The documentation is available at `https://keras.io`. Keras does not handle itself low-level operations such as tensor manipulation and differentiation. Instead, it relies on a specialized tensor library to do so. That library serves as the **backend** engine of Keras. Keras was organized in a modular way, so several different backend engines could be plugged seamlessly into Keras. Initially, it worked with three backend implementations, TensorFlow, Theano and CNTK, but the last two options have been dropped in the recent versions, so Keras is no longer multi-backend.

Just to give you an idea why Keras is popular, it has been said that the number of keystrokes needed to specify a deep learning model in Keras is one half of what was needed in old TensorFlow. So, the leading choice of deep learning aficionados has been, for years, be the combination of TensorFlow (backend) and Keras (frontend). Pythonistas can have this combo in two ways. (a) using the package `keras` with the TensorFlow backend, or (b) using the module `keras` of the package `tensorflow`. we use the second option in this course. it is available yo you just running `pip install tensorflow` in the shell. If you have problems with the installation (this was happening to students of this course until recently), you can switch to Google Colab for your deep learning experience.

*Note*. Last news are: (a) Google seems to be replacing TensorFlow by a new thing, called JAX (see `https://jax.readthedocs.io`), (b) Keras is going to be multi-backend again, with optional backeneds being TensorFlow, PyTorch and JAX. Stay tuned! 

## MLP networks in Keras

Let us suppose, in this section, that you wish to train a MLP classifier using `tensorflow.keras`. For the examples discussed in this course, in which the target vector is numeric, you will have enough with the modules `models` and `layers`, which you can import as:

```
from tensorflow.keras import models, layers
```

The module `models` has two classes, `.Sequential()` and `.Model()`. The first one is enough for most of us. The other class, known as the **Functional API**, is used with more sophisticated architectures. 

A simple way to specify the network architecture is to create a list of layers. The layers are extracted from classes of the module `layers`. For a MLP network we only need the class `Dense()`. For instance, a MLP network with one hidden layer of 32 nodes for the MNIST data would be specified as:

```
network = [layers.Dense(32, activation='relu'), layers.Dense(10, activation='softmax')]
```

You start by initializing the class, with the default specification:

```
clf = models.Sequential(network)
```

Then, the model is compiled, with the method `.compile()`:

```
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
```

Now, we can apply the method `.fit()`, which is just a bit more complex than in scikit-learn. Assuming that you have previously performed a train/test split, this step could be:

```
clf.fit(X_train, y, epochs=10)
```

Note that, the number of iterations (the parameter `epochs`) is specified as an argument of `.fit()`, not as in scikit-learn, when instantiating the estimator. In `tensorflow.keras`, you can run `.fit()` many times, getting a gradual improvement.

The method `.fit()` prints a report tracking the training process. You can stop this with the argument `verbose=0`. After fitting, we validate the model on the test set:

```
clf.evaluate(X_test, y_test)
```

Here, the method `.predict()` returns the class probabilities (not the predicted class), just as the method `.predict_proba()` in scikit-learn.

## Deep learning application to computer vision

A **convolutional neural network** (CNN) is a regularized version of a MLP network. In the classic MLP network, input and hidden layers were dense, that is, every node was connected to all neurons in the next layer. On the contrary, CNN's have low connectivity, and connections are selected according a design which takes advantage of the hierarchical pattern in the data and assemble complex patterns using smaller and simpler patterns. The fundamental difference between a dense layer and a convolution layer is that dense layers learn global patterns in their input feature space (*e.g*. for a MNIST digit, patterns involving all pixels), while convolution layers learn local patterns, *i.e*. patterns found in small 1D or 2D windows of the inputs.

There are two subtypes of convolutional networks:

* **1D convolutional networks** (Conv1D), used with sequence data (see below).

* **2D convolutional networks** (Conv2D), used in image classification. 

In the CNN's used in **image classification**, the input is a 3D tensor, called a **feature map**. The feature map has two spatial axes, called **height** and **width**, and a **depth** axis. For a RGB image, the dimension of the depth axis would be 3, since the image has 3 color channels, red, green, and blue. For black and white pictures like the MNIST digits, it is just 1 (gray levels).

A convolution layer extracts patches from its input feature map, typically with a 3 $\times$ 3 window) and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a 3D tensor: it has width, height and depth. Its depth can be arbitrary, since the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors like in an RGB input, rather they stand for what we call **filters**. The filters encode specific aspects of the input data. For instance, at a high level, a single filter could be encoding the concept "presence of a face in the input".

Practitioners typically use two strategies for extracting more of their data:

* **Transfer learning**. Instead of starting to train your model with random coefficients, you start with those of a model which has been pre-trained with other data. There are many options for that, among the classification algorithms that have been trained the **ImageNet** database (see `image-net.org`).

* Expanding the training data with images obtained by transforming the original images. Typical transformations are: rotation with a random angle, random shift and zoom.

## Applications to sequence data

The second area of success of deep learning is **sequence data**. This is a generic expression including text (sequences of words or characters), time series data, video and others. Although I do not have room here for this type of data, let me mention that the main types of networks used in this context are:

* 1D convolutional networks, with applications to machine translation, document classification and spelling correction.

* **Recurrent neural networks** (RNN), with applications to handwritting and speech recognition, sentiment analysis, video classification and time series data.

* Networks with **embedding layers**, with applications to natural language processing (NLP) and recommendation systems.

But the use of CNN and RNN models with text data may get obsolete very soon, given the strong push recently given by generative AI in this field. Though this is beyond the scope of this course, it is good to know that it is, precisely, the hottest are right now.

## CNN models in Keras

Let us use again the MNIST data as to illustrate the Keras syntax, now for CNN models. The height and the width are 28, and the depth is 1. We start by reshaping the training and test feature matrices as 3D arrays, so they can provide inputs for a `Conv2D` network:

```
X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
```

*Note*. This is reshaping may not be needed if you get the MINST data from other sources than the GitHub repository that is used in this course. 

The network architecture can be specified, in a comprehensive way, as a list of layers. The following list is quite typical:

```
network = [layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')]
```

The first layer is a Conv2D layer of 32 nodes. Every node takes data from a 3 $\times$ 3 window (submatrix) of the 28 $\times$ 28 pixel matrix, performing a convolution operation on those data. There are 26 $\times$ 26 such windows, so the output feature map will have height and width 26. The convolution is a linear function of the input data. For a specific node, the coefficients used by the convolution are the same for all windows.

`Conv2D` layers are typically alternated with MaxPooling2D layers. These layers also use windows (here 2 $\times$ 2 windows), from which they extract the maximum value. In the MaxPooling2D layer, the windows are disjoint, so the size of the feature map is halved. Therefore, the output feature map will have height and width 13.  

We continue with two Conv2D layers, with 64 nodes each, with a MaxPooling2D layer in-between. The output is now a tensor of shape (3, 3, 64). 

The network is closed by by a stack of two Dense layers. Since the input in these layers has to be a vector, we have to flatten the 3D output of the last Conv2D layer to a 1D tensor. This is done with a Flatten layer, which involves no calculation, but just a reshape. 

Next, we initialize the class `Sequential()`, in order to specify the network architecture:

```
clf = models.Sequential(network)
```

Once the network has been completely specified, we apply, as in the MLP example, the methods `compile`, `fit` and `evaluate`:

```
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
clf.fit(X_train, y_train, epochs=10)
clf.evaluate(X_test, y_test) 
```

Alternatively, you can fit and evaluate the model in one shot, testing after every epoch:

```
clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

Once the model has been compiled, you can also print a summary of the network as:

```
clf.summary()
```
