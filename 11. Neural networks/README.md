# 11. Neural networks

### What is a neural network?

An (artificial) **neural network** is an interconnected set of computational elements, called **nodes** or neurons, organized in **layers**. Every connection of a node to another node has a **weight**. In machine learning, these weights are learned from training data.

There are many types of neural networks, but this course only covers the **multilayer perceptron** (MLP). Although the field of neural networks is very rich and diverse (see Haykin, 1999, for a general presentation), practitioners usually refer to the multilayer perceptron when using the expressions neural network or artificial neural network (ANN).

Under the hood, a MLP model is just a set of equations, as explained below. MLP models can be used for both regression and classification. In scikit-learn, they are available from the classes `MLPRegressor` and `MLPClassifier` of the subpackage `neural_networks`.

A multilayer perceptron is formed by:

* The **input layer**, whose nodes are the features used for the prediction.

* The **output layer**. In regression, it has a unique node, which is the target (as in the figure below). In classification, it has one node for every target value.

* Some **hidden layers**. If the network is **fully-connected**, that is, if every node of a layer is connected to all the nodes of the following layer, the model is completely specified by the number of hidden layers and the number of nodes in each hidden layer. This course only covers fully-connected networks.

How do these networks work? Let us see first what happens at a hidden node. Suppose that *Z* is a hidden node and *U1*, *U2*, …, *Uk* are the nodes of the preceding layer. Then, the values of *Z* are calculated as

<img src="https://render.githubusercontent.com/render/math?math=\large Z = F\big(w_0 + w_1 U_1 %2B w_2 U_2 %2B \cdots %2B w_k U_k\big).">

The slope coefficients *w1*, *w2*, …, *wk* are called weights, and the intercept *w0* is called **bias**. *F* is the **activation function**. The role of the activation function is to introduce nonlinearity in the model. A bit of mathematical detail is given below.

The multilayer perceptron could be seen as if the samples were circulating through the network one-by-one. The feature values are entered in the input nodes, which send them to the nodes of the first hidden layer. At each hidden node, they are combined using the corresponding weights, and the result is transformed by means of the activation function. The hidden nodes send the resulting values to the nodes of the next layer, where they are combined. According to the legend, this simulates how animal neurons learn.

![](https://github.com/cinnData/MLearning/blob/main/11.%20Neural%20networks/fig%2011.1.png)

The model of the abobe figure is a MLP regressor with one hidden layer of two nodes. But this is just a representation of a set of three equations. The two equations that allow us to go from the input layer to the hidden layer combine the features with weights *w1A*, *w2A* and *w3A* and *w1B*, *w2B* and *w3B*, respectively. The biases are *w0A* and *w0B*, respectively.

At the hidden nodes *A* and *B*, activation is applied to the values given by these equations. Once the activation has been applied, *A* and *B* are combined in the third equation with weights *wAY* and *wBY* and bias *w0Y*, to obtain the predicted value of *Y*. This model has a total of 11 parameters.

### The activation function

The choice of the activation function is based on performance, since we do not have any serious theory which could explain why this mathematical formula is better than that one. Just a few years ago, the **logistic function** was the recommended activation function in the hidden layers, although some preferred a similar formula called the **hyperbolic tangent function**. The current trend favors the **rectified linear unit function** (ReLU), which is the default in scikit-learn. ReLU(*x*) is equal to *x* when *x* > 0 and equal to 0 otherwise. So, if you accept the default of scikit-learn, the activation in the hidden layers consists in turning the negative incoming values into zeros.

In a MLP regressor (as in Figure 1), there is no activation at the (single) output node, so the equation predicting the values at that node is linear. In a MLP classifier, there are as many output nodes as target values. An activation function called the **softmax function** is applied to the whole set of incoming values, turning them into a set of **class probabilities**. The mathematical expressions involved in the definition of the softmax function are similar to the logistic function formula.

### Other technicalities

* *The number of hidden layers and nodes*. It is recommended to start with something small and let it grow if that is supported by an improved performance. Many practitioners (including myself) use powers of two (8, 16, 32, etc), although there is no theory behind that.

*  *How to find the optimal weights*. This typically follows a method called **backpropagation**. Initially, the weights are randomly assigned. Then, an iterative process starts. At every step, the prediction is performed with the current weights, the value of a **loss function** is calculated and the weights are adjusted in order to reduce the loss. The process is expected to converge to an optimal solution, but, in practice, a maximum number of iterations is prespecified. In regression, the loss is the sum of the squared errors, while, in classification, it is given by a formula from information theory called the **cross-entropy**.

* *The optimization method*, called the **solver** in scikit-learner. The **limited-memory Broyden-Fletcher-Goldfarb-Shanno method** (LBFGS) has been the choice for many years, but the current default option is the **stochastic gradient descent** (SGD). For small data sets, however, LFBGS can converge faster and perform better.

* *The learning rate* is a parameter which controls how fast the adjustment of the weights is done. If it is too low, there is no convergence to the optimal solution. If it is too high, you can overshoot the optimal solution. Modern ML software allows setting an initial learning rate and decrease it as the learning process goes on. 
 
* *The batch size*. In the SGD method, the training data are randomly partitioned in batches in every iteration. The batches are tried one-by-one and the weights are modified for every batch.

* *Normalization*. The multilayer perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. In the old data mining suites, normalization was applied as a part of the algorithm, and the output was scaled back to the original range. It is not so in scikit-learn.

### MLP models in scikit-learn

In scikit-learn, MLP models  are available in the classes `MLPRegressor` and `LPClassifier`, from the subpackage `neural_network`. The key parameter is `hidden_layer_sizes`, which sets the number of nodes of each hidden layer.

An example of a MLP classifier with a single hidden layer of 32 nodes would be:

`from sklearn.neural_network import MLPClassifier`

`mlpclf = MLPClassifier(hidden_layer_sizes=(32))`

Here is an alternative layout with two layers of 16 nodes each:

`mlpclf = MLPClassifier(hidden_layer_sizes=(16,16))`

The default is one layer of 100 hidden nodes, but I would recommend you to start with something smaller. The default solver is `solver='adam'`, which uses a variant of the SGD method, with a constant learning rate of 0.001 and a batch size of 200. But you better leave that as it is until you are familiar with the technical stuff.

The default for the maximum number of iterations is `max_iter=200`. But neural network models are prone to overfitting. So a common practice is to stop the iterations before achieving the convergence, since after a certain number of iterations (frequently no more than 50) the performance improves on the training data, but not on the test data. The code would be:

`mlpclf = MLPClassifier(hidden_layer_sizes=(32), max_iter=10)`

The methods `fit`, `predict` and `score` work as usual in supervised learning.

### Normalization

In scikit-learn, normalization is left to the user. You will probably need it. A method called **min-max normalization** is frequently applied in this context. In min-max normalization, the features are forced, through a linear transformation, into the 0-1 range. The formula for this transformation is

<img src="https://render.githubusercontent.com/render/math?math=\large Z = \displaystyle\frac{X-\min(X)}{\max(X)-\min(X)}\,.">

A transformer of the class `MinMaxScaler`, from the subpackage `preprocessing`, can be used to create a new feature matrix in which every column is normalized. It is similar to other preprocessing methods. You can instantiate it with:

`from sklearn.preprocessing import MinMaxScaler`

`scaler = MinMaxScaler()`

Next, you apply `fit` and `transform` to the original feature matrix `X`, to obtain the normalized matrix `Z`:

`scaler.fit(X)`

`Z = scaler.transform(X)`
