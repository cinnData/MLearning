# [ML-01] What is machine learning?

## Machine learning

**Machine learning** (ML) is a branch of **artificial intelligence** (AI). You may have heard about other branches, such as robotics, or speech recognition. The objective of machine learning is the development and implementation of **algorithms** which learn from data how to accomplish difficult or tiring tasks.

In general, an algorithm is a set of rules that precisely define a sequence of operations. In the context of machine learning, it is the set of instructions a computer executes to learn from data. This process is called **training**, and we say that we have developed a **model**. For instance, a model which classifies the potential customers of a lending institution as good or bad creditors.

The model learnt from the **training data** is ofetn tested on different data, which are then called **test data**. This is **model validation**. Validation is needed for models whose complexity allows them to overfit the data. **Overfitting** happens when the performance of a model on fresh data is significantly worse than its performance on the training data. Overfitting is a fact of life for many machine learning algorithms, eg for those used to develop neural network models. So, validation must be integrated in the development of many models.

## Supervised and unsupervised learning

In machine learning, based on the structure of the data used in the training process, it is usual to distinguish between supervised and unsupervised learning. Roughly speaking, **supervised learning** is what the statisticians call prediction, that is, the description of one variable ($Y$), in terms of other variables (the $X$'s). In the ML context, $Y$ is called the **target**, and the $X$'s are called **features**. The units (they can be customers, products, etc) on which the features and the target are observed are called **samples** (this term has a different meaning in statistics).

The term **regression** applies to the prediction of a (more or less continuous) numeric target, and the term **classification** to the prediction of a categorical target. In **binary classification**, there are only two target values or **classes**, while, in **multi-class classification**, there can be three or more. The classification model predicts a probability for every class.

In an example of regression, we may try to predict the price of a house from a set of attributes of that house. In one of classification, whether a customer is going to quit our company, from his/her demographics plus some measures of customer activity.

In **unsupervised leaning**, there is no target to be predicted (only $X$'s). The objective is to learn patterns from the data. Unsupervised learning is more difficult, and more creative, than supervised learning. The two classics of unsupervised learning are **clustering**, which consists in grouping objects based on their similarity, and **association rules** mining, which consists in extracting from the data rules such as *if A, then B*. A typical application of clustering in business is **customer segmentation**. Association rules are applied in **market basket analysis**, to associate products that are purchased (or viewed in a website) together. Other relevant examples of unsupervised learning are **dimensionality reduction** and **anomaly detection**.

## Variations

In a classification context, distinction is frequently made between labeled and unlabeled data. The **labels** are the target values. When the data come with labels, one can use supervised learning and, when they are not, unsupervised learning. For instance, in image classification, the data ususally consist in a set of pictures. The pictures can be human-labeled, which makes the training data more expensive, or unlabeled.

In-between supervised and unsupervised learning, we have **semi-supervised learning**, which combines supervised and unsupervised learning, requiring only a small portion of the learning data be labeled. An alternative approach is **self-supervised learning**, which uses only unlabeled data. A well known example of self-supervised learning is Google's **word2vec**, a technique which learns word associations to generate a representation of words as vectors in a multidimensional space. This representation is used later in other jobs.

Another variation is **reinforcement learning**, which is one of the current trending ML topics, because of its unexpected success in playing games like go and StarCraft II. It is not considered as supervised nor as unsupervised learning. For more information on this, see Mitchell (2020).

From the point of view of the practical implementation, we can also distinguish between batch and on-line learning. In **batch learning**, the algorithm is trained and tested on given data sets and applied for some time without modification. In **on-line training**, it is continuously retrained with the incoming data. The choice between batch and continuous learning depends on practical issues, rather than on theoretical arguments.

## References

1. E Alpaydin (2016), *Machine Learning*, MIT Press.

2. P Domingos (2015), *The Master Algorithm*, Basic Books.

3. M Mitchell (2020), *Artificial Intelligence: A Guide for Thinking Humans*, Pelican.
