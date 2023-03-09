#  [ML-10] Neural networks

## What is a neural network?

An (artificial) **neural network** is an interconnected set of computational elements, called **nodes** or neurons, organized in layers. Every connection of a node to another node has a **weight**. In machine learning, these weights are learned from the training data.

There are many types of neural networks. I start with the **multilayer perceptron** (MLP), which has been the standard approach for many years, before **deep learning** rose to prominence. Practitioners usually referred to the multilayer perceptron when using the expressions neural network or artificial neural network (ANN).

Under the hood, a MLP model is just a set of equations, as explained below. MLP models can be used for both regression and classification. In scikit-learn, they can be extracted from the classes `MLPRegressor` and `MLPClassifier` of the subpackage `neural_networks`. In this lecture I skip this, presenting directly the implementation of the package **Keras**.

## The basics of the MLP model

A multilayer perceptron is formed by:

* The **input layer**, whose nodes are the features used for the prediction.

* The **output layer**. In regression models, it has a unique node, which is the target (as in the figure below). In classification models, one node for every target value.

* A sequence of **hidden layers** placed between the input and the output layers. If the network is **fully-connected**, that is, if every node of a layer is connected to all the nodes of the following layer, the model is completely specified by the number of hidden layers and the number of nodes in each hidden layer.

How do these networks work? Suppose first that $Z$ is a hidden node and $U_1, U_2, \dots, U_k$ are the nodes of the preceding layer. Then, the values of $Z$ are calculated as

$$Z = F\big(w_0 + w_1U_1 + w_2U_2 + \cdots + w_kU_k\big).$$

In this context, the slope coefficients $w_1, w_2, \dots, w_k$  are called weights, and the intercept $w_0$ is called **bias**. $F$ is the **activation function**. The role of the activation function is to introduce nonlinearity in the model. A bit of mathematical detail is given below.

The multilayer perceptron could be seen as if the samples were circulating through the network one-by-one. The feature values are entered in the input nodes, which send them to the nodes of the first hidden layer. At each hidden node, they are combined using the corresponding weights, and the result is transformed by means of the activation function. The hidden nodes send the resulting values to the nodes of the next layer, where they are combined. According to the legend, this copies the way  how animal neurons learn.

Let me help your intuition with the graphical representation of a small network. The model of the figure below is a MLP regressor with one hidden layer of two nodes. The diagram is just a graphical representation of a set of three equations. The two equations that allow us to go from the input layer to the hidden layer combine the features with weights $w_{1A}$, $w_{2A}$ and $w_{3A}$, and $w_{1B}$, $w_{2B}$ and $w_{3B}$, respectively. The biases are $w_{0A}$ and $w_{0B}$, respectively.

![](https://github.com/cinnData/MLearning/blob/main/Figures/fig%_7.1.png)

At the hidden nodes  and , the activation function is applied to the values given by these equations. Once the activation has been applied,  and  are combined in the third equation with weights  and  and bias , to obtain the predicted value of . This model has a total of 11 parameters.

The activation function
The choice of the activation function is based on performance, since we do not have any serious theory which could explain why a specific mathematical formula works better than others. Just a few years ago, the logistic function was the recommended activation function in the hidden layers, although some preferred a similar formula called the hyperbolic tangent function. The current trend favors the rectified linear unit function (ReLU), which is the default in scikit-learn and Keras.  is equal to  when  and equal to 0 otherwise. So, if you accept this default, the activation in the hidden layers consists in turning the negative incoming values into zeros.
In a MLP regressor (as in the figure), there is no activation at the (single) output node, so the equation predicting the values at that node is linear. In a MLP classifier, there are as many output nodes as target values. An activation function called the softmax function is applied to the whole set of incoming values, turning them into a set of class probabilities. The mathematical expressions involved in the definition of the softmax function are similar to the logistic function formula.
Other technicalities
How to find the optimal weights. Initially, the weights are randomly assigned. Then, an iterative process starts. At every step, the prediction is performed with the current weights, the value of a loss function is calculated, and the weights are adjusted in order to reduce the loss. The process is expected to converge to an optimal solution, but, in practice, a maximum number of iterations is pre-specified. In regression, the loss is usually the MSE, while, in classification, it is the cross-entropy. Experts may suggest to change this for specific jobs, but beginners stick to the common choices. The adjustment of the weights starts at the last layer, and continues backwards until the input layer. This is called backpropagation. 
The optimization method, called solver in scikit-learn and optimizer in Keras. The current trend favors the stochastic gradient descent (SGD), which has many variants. The Keras standard is optimizer='rmsprop'.
The learning rate is a parameter which controls how fast the adjustment of the weights is done. If it is too low, there is no convergence to the optimal solution. If it is too high, you can overshoot the optimal solution. Modern ML software allows setting an initial learning rate and decrease it as the learning process goes on. The Keras default is 0.001.
The batch size. In the SGD method, the training data are randomly partitioned in batches in every iteration. The batches are tried one-by-one and the weights are modified every time that a batch is tried. The Keras  default is 32, though 64 is quite popular. 
Normalization. The multilayer perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. In the old data mining suites, normalization was applied as a part of the algorithm, and the output was scaled back to the original range. It is not so in Python ML toolbox.
Normalization
In scikit-learn and Keras, normalization is left to the user. A method called min-max normalization is frequently applied in the neural network context. In min-max normalization, the features are forced, through a linear transformation, into the 0-1 range. The formula for this transformation is

MinMaxScaler, a transformer class from the subpackage preprocessing, can be used to create a new feature matrix in which every column is normalized. It is similar to other preprocessing tools. You can instantiate a transformer with:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

Then, you apply the methods fit and transform to the original feature matrix X, to obtain the normalized feature matrix Z:

scaler.fit(X)
Z = scaler.transform(X)
What is deep learning?
Deep learning, the current star of machine learning, is based on neural networks. The success of deep learning, not yet fully understood, is attributed to the ability of creating improved representations of the input data by means of successive layers of features. 
Under this perspective, deep learning is a successful approach to feature engineering. Why is this needed? Because, in many cases, the available features do not provide an adequate representation of the data, so replacing the original features by a new set may be useful. At the price of oversimplifying a complex question, the following two examples may help to understand this:
A model for predicting the sale price of a house from features like the square footage of the plot and the house, the location, the number of bedrooms, the existence of a garage, etc. You will probably agree that these are, indeed, the features that determine the price, so they provide a good representation of the data, and a shallow learning model, such as a random forest regressor, would be a good approach. No feature engineering is needed here.
A model for image classification. Here, the available features are related to a grid of pixels. But we do not classify an image based on specific pixel positions. Recognition is based on shapes and corners. A shape is a created by a collection of pixels, each of them close to the preceding one. And a corner is created by tho shapes intersecting in a specific way. So, we have the input layer of pixels, a first hidden layer of shapes providing a better representation, a second layer of corners providing an even better representation.
The number of hidden layers in a neural network is called the depth. But, although deep learning is based on neural networks with more than one hidden layer, there is more in deep learning than additional layers. In the MLP model as we have seen it, every hidden node is connected to all the nodes of the preceding layer and all the nodes of the following layer. In the deep learning context, these fully-connected layers are called dense. But there are other types of layers, and the most glamorous applications of deep learning are based on networks which are not fully-connected.
TensorFlow and Keras
The mathematics involved in the specification of the neural networks used in deep learning are written in terms of tensors. A tensor is the same as what we call array in NumPy. So, a 0D tensor is a scalar (a number), a 1D tensor is a vector, a 2D tensor is a matrix, etc. The operations with tensors and their properties are the object of tensor algebra, which is an extension of the matrix algebra which you may have learnt in undergraduate courses.
You do not need to worry about the math technicalities, as far as you pay attention to the specification of the shape of the input and output tensors in every layer of the network. Among the many attempts to implement the mathematics of the deep learning networks, including the tensors, the top popular one is TensorFlow, developed at Google Brain and released in 2015. The main competitor is PyTorch, released by Facebook. Recent news are that PyTorch is becoming the top choice in research (academic journals and conferences), although TensorFlow is still the undisputed leader in industry (according to job listings).
Keras is a deep learning framework for Python (there is also a version for R), which provides a convenient way to define and train deep learning models. The documentation is available at keras.io. Keras does not handle itself low-level operations such as tensor manipulation and differentiation. Instead, it relies on a specialized tensor library to do so. That library serves as the backend engine of Keras. Keras is organized in a modular way, so several different backend engines can be plugged seamlessly into Keras. Currently, the three existing backend implementations are the TensorFlow backend, the Theano backend, and the CNTK backend.
The leading choice of deep learning aficionados, nowadays, seems to be the combination of TensorFlow (backend) and Keras (frontend). Pythonistas can have this combo in two ways. (a) using the keras package with the default backend, which is TensorFlow, so they do not have to specify the backend, or (b) using the module keras of the package tensorflow. For the first option, you can help yourself with Chollet (2017) or the Keras website. For the second option, use Géron (2017), or the TensorFlow API documentation site (tensorflow.org/api_docs/python/tf). The rest of this note refers to the first option, for which you only have to install Keras. By installing Keras, TensorFlow is installed on the fly. Just to give you an idea why Keras is popular, it has been said that the number of keystrokes needed to specify a deep learning model in Keras is one half of what was needed in old TensorFlow.
Note. TensorFlow is the only backend available in recent versions of Keras. Since there seems to exist a conflict between TensorFlow and M1 chips, you may have to pick an old version of Keras for your new Mac.
MLP networks in Keras
Let me suppose, to keep it short, that you wish to train a MLP classifier using Keras. For the examples discussed in this course, in which the target vector is numeric, you will have enough with the subpackages models and layers, which you can import as:

from keras import models, layers

The models module has two classes, Sequential and Model. The first one, in which we add layers in a sequential way with the method add, is enough for most of us. The other class, known as the Functional API, is used with more sophisticated architectures. You start by initializing the class, with the default specification:

network = models.Sequential()

A layer is extracted from a class of the subpackage layers. If you only use the class Dense, you get a MLP network. For instance, a MLP network with one hidden layer of 32 nodes for the MNIST data would be specified as:

network.add(layers.Dense(32, activation='relu', input_shape=(784,)))
network.add(layers.Dense(10, activation='softmax'))

Note that, in spite of using the default activation functions, I specify them, for clarity. Once the network architecture is completely specified, the model is compiled, with the method compile:

network.compile(optimizer='rmsprop', loss=‘sparse_categorical_crossentropy',
    metrics=['accuracy'])

Finally, we apply the method fit, which is just a bit more complex than in scikit-learn. Assuming that you have previously performed a train/test split, this step could be:

network.fit(X_train, y, epochs=10, batch_size=64)

The epochs are the iterations, that is, the number of times every sample passes through the network (max_iter in scikit-learn). This number is typically low, to prevent overfitting. The default is epochs=1. The samples don’t pass all at once, but in random batches. The batch size is typically set as a power of 2, the default being 32, though 64 is more popular.
fit returns a report containing information on the accuracies obtained in every epoch. Then, you validate the model on the test set:

network.evaluate(X_test, y_test)

Eventually, you can add a network.predict() line, which, in Keras, gives the class probabilities.
Deep learning application to computer vision
A convolutional neural network (CNN) is a regularized version of a MLP network. In a MLP network, each neuron in one layer is connected to all neurons in the next layer. CNN's have low connectivity, and connections are selected according a design which takes advantage of the hierarchical pattern in the data and assemble complex patterns using smaller and simpler patterns. The fundamental difference between a dense layer and a convolution layer is that dense layers learn global patterns in their input feature space (e.g. for a MNIST digit, patterns involving all pixels), while convolution layers learn local patterns, i.e. patterns found in small 1D or 2D windows of the inputs.
There are two subtypes of convolutional networks:
1D convolutional networks (Conv1D), used with sequence data (see below).
2D convolutional networks (Conv2D), used in image classification, recommendation systems and natural language processing (NLP). 
In the CNN's used in image classification, the input is a 3D tensor, called a feature map. The feature map has two spatial axes, called height and width, and a depth axis. For a RGB image, the dimension of the depth axis would be 3, since the image has 3 color channels, red, green, and blue. For black and white pictures like the MNIST digits, it is just 1 (gray levels).
A convolution layer extracts patches from its input feature map, and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a 3D tensor: it has width, height and depth. Its depth can be arbitrary, since the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors like in an RGB input, rather they stand for what we call filters. The filters encode specific aspects of the input data. For instance, at a high level, a single filter could be encoding the concept "presence of a face in the input".
Practitioners typically use two strategies for extracting more of their data:
Transfer learning. Instead of starting to train your model with random coefficients, you start with those of a model which has been pre-trained with other data. There are many options for that among the classification algorithms that have been trained the ImageNet database (see image-net.org).
Expanding the training data with images obtained by transforming the original images. Typical transformations are: rotation with a random angle, random shift and zoom.
Applications to sequence data
The second area of success of deep learning is sequence data. This is a generic expression including text (sequences of words or characters), time series data, video and others. Although I do not have room here for this type of data, let me mention that the main types of networks used in this context are:
1D convolutional networks, with applications to machine translation, document classification and spelling correction.
Recurrent neural networks (RNN), with applications to handwritting and speech recognition, sentiment analysis, video classification and time series data.
Networks with embedding layers, with applications to recommendation systems.
CNN models in Keras
Let me use again the MNIST data as to illustrate the Keras syntax, now for CNN models. The height and the width are 28, and the depth is 1. We start by reshaping the training and test feature matrices as 3D arrays, so they can provide inputs for a Conv2D network:

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

Next, we initialize the class Sequential, in order to specify the network architecture:
network = models.Sequential()

The network is built layer by layer. The first layer is a Conv2D layer of 32 nodes. Every node takes data from a  window (submatrix) of the  pixel matrix, performing a convolution operation on those data. There are  such windows, so the output feature map will have height and width 26. The convolution is a linear function of the input data. For a specific node, the coefficients used by the convolution are the same for all windows.

network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

Conv2D layers are typically alternated with MaxPooling2D layers. These layers also use windows (here  windows), from which they extract the maximum value. In the MaxPooling2D layer, the windows are disjoint, so the size of the feature map is halved. Therefore, the output feature map will have height and width 13.  

network.add(layers.MaxPooling2D((2, 2)))

We continue with two Conv2D layers, now of 64 nodes, with a MaxPooling2D layer in-between. The output is now a tensor of shape (3, 3, 64). 

network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))

The network is closed by by a stack of two Dense layers. Since the input in these layers has to be a vector, we have to flatten the 3D output of the last Conv2D layer to 1D tensor. This is done with a Flatten layer, which involves no calculation, but just a reshape. 

network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

You can print a summary of the network as:

network.summary()

Once the network has been completely specified, we apply, as in the MLP example, the methods compile, fit and evaluate:

network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy’,
    metrics=[‘accuracy'])
network.fit(X_train, y_train, epochs=10, batch_size=64)
network.evaluate(X_test, y_test) 

Alternatively, you can fit and evaluate the model in one shot, testing after every epoch:

network.fit(X_train, y_train, epochs=10, batch_size=64,
    validation_data=(X_test, y_test))

