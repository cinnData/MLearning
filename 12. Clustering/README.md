# 12. Clustering

### What is a clustering algorithm?

A **clustering algorithm** groups samples into **clusters**, based on their similarity. Clustering methods have been applied for a long time in many fields, under specific names. For instance, in marketing, clustering customers is called **customer segmentation**. In the machine learning context, clustering it is regarded as **unsupervised learning**.

There are many clustering methods, all based on the same principle, to get the maximum similarity within clusters and the minimum similarity between clusters. This is operationalized through a **similarity measure**. Similarity measures are also used in other machine learning techniques, such as the *k*-**nearest neighbor** (kNN) algorithm and **collaborative filtering**.

There are, basically, two approaches to clustering: the **distance-based** methods, such as the *k*-means algorithm, and the **probability-based** methods, such as the EM clustering algorithm. Only the *k*-means algorithm is considered here, because most of the other methods, in spite of their popularity in textbooks, have **scalability** problems, meaning that they do not work, or become too slow, with big data sets.

A warning note: clustering algorithms always produce clusters. But the clusters you get could be useless for their intended application. For instance, if you expect them to help to understand your customers, they have to be described in a intelligible way. This would probably imply a low number of clusters. 

Frequently, professors and teaching materials suggest that, for a clustering to be useful, the number of clusters have to be small. This is may be true when the clusters are intended to be managed by humans, which is not true in many business applications. A big e-retailer like Amazon can easily manage hundreds of clusters, with no human mind understanding what they are. In some cases, the clusters are used only to speed up computation. For instance, to get recommendations faster, you can search them within clusters of products. 

### Similarities

In the distance-based methods, the similarity of two instances is measured by a distance formula, which is usually the **Euclidean distance**. The Euclidean distance is the ordinary, real-life distance. The following example illustrates this.

Pick two points in the plane, such as *x* = (-1, 3) and *y* = (2, 2). The vector with origin *x* and endpoint *y* is (3, -1). The distance between *x* and *y* is the length of this vector:

<img src="https://render.githubusercontent.com/render/math?math=\large \sqrt{3^2 %2B 5^2} = 7.211.">

The general formula, for a *p*-dimensional space, is

<img src="https://render.githubusercontent.com/render/math?math=\large \textrm{dist}(x,y) = \sqrt{(x_1-y_1 )^2 %2B \cdots %2B (x_p-y_p)^2}.">

This formula can be applied to any pair of rows of a data set with *p* numeric columns. In the machine learning toolbox, the Euclidean distance is the default similarity measure. Nevertheless, in particular contexts, such as **text mining**, other measures, like the **cosine-based similarity**, are preferred.

It is not rare, in real data, that some features show a much higher variation than the rest. Formulas like the Euclidean distance make those features too influential on the clustering process. To prevent this, the features involved in the clustering process can be normalized. Min-max normalization is typical in this context. Beware that normalization can change significantly your clusters. In customer segmentation, for instance, this is a relevant issue.

### Cluster centers

Suppose that you wish to group the samples in *k* clusters using $p$ numeric features. Many clustering methods are based on finding a set of *k* points in the *p*-dimensional space of the features, called **centers**, and clustering the samples around the centers. Every sample will be assigned to the cluster whose center is most similar. Typically, the similarity is the Euclidean distance. 

The centers can also be used for assigning a cluster to a new sample which has not used to find the centers. We just select the cluster whose center is closer to that new sample.

In real-world applications, we look at the center as an artificial sample which we consider as the "typical element" of the cluster. The values that this artificial sample takes for the different features are used to create a description of the cluster, as far as this makes sense. This is the typical approach in customer segmentation. So a marketing manager may say that he has a segment of customers above 60, with annual family income between $100,000 and $250,000, who frequently watch soap opera TV comedy series. This is just a description of the center of that segment.

The methods that use this approach differ on how the centers are extracted from the training data. The next section presents a brief discussion of the top popular one, the *k*-means clustering algorithm, and some of ideas on how you can decide the number of centers *k*.

### *k*-means clustering

The *k*-**means algorithm** searches for a set of $k$ centers such that the corresponding clusters have any of the mathematically equivalent properties:

* The center is the average of the samples of the cluster.

* The sum of the squared (Euclidean) distances of the samples to the centers of their respective clusters is minimum.

The *k*-means search is iterative. The steps are:

* A random choice of $k$ samples is taken as the initial set of centers.

* Every sample is assigned to the cluster whose center is the closest one (in the Euclidean distance).

* Then, the average of every cluster is taken as the new center and the samples are reassigned based on the new centers.

* This is iterated until a prespecified stopping criterion is met.

* The algorithm returns the collection of centers and a vector with the cluster labels for the training samples.

Despite some drawbacks, *k*-means remains the most widely used clustering algorithm. It is simple, easily understandable and reasonably scalable, and it can be easily modified to deal with streaming data. 

In *k*-means clustering, you have to specify the number of clusters *k*. Even if this is something on which you do not have a definite number, you will probably have a preliminary idea, so you will work around that. For instance, you may wish to have a number of clusters from 3 to 6. You will then try *k* = 3, 4, 5 and 6, comparing the results. You will consider the cluster sizes, since you do not want clusters which are too small, and you will monitor how the clusters change when you increase the number of clusters. 

Stability is expected from a respectable segmentation. Fisrt, mind that, due to the random start, two runs of the *k*-means clustering can give different results. The difference should not be relevant. Also, you can apply here a out-of-sample for testing your clusters. You can leave out a test data subset, group the remaining samples again, and compare the partitions. 

### Clustering in scikit-learn

In scikit-learn, *k*-means clustering is provided by the class `KMeans` from the subpackage `cluster`. Suppose that your choice is *k* = 4. You start as usual in scikit-learn, by instatiating your estimator. Since the default number of clusters is `n_clusters=8`, you have to change that: 

`from sklearn.cluster import KMeans`

`clus = KMeans(n_clusters=4)`

The method `fit` is applied to a feature matrix as usual:

`clus.fit(X)`

There are two interesting attributes here. `labels_` is a 1d array containing a cluster label for every sample. With `n_clusters=4`, the labels would be 0, 1, 2 and 3. 

`labels = clus.labels_`

The attribute `cluster_centers_` gives the coordinates of the cluster centers. In this case, this would be a matrix with four rows, one for every center. For every row, the terms of that row would be the mean values of the features on the corresponding cluster.

`centers = clus.cluster_centers_`

The examination of the matrix `centers` can help you to describe the clusters. In some cases, you will normalize the feature matrix. 
