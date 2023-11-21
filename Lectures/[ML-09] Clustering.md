# [ML-09] Clustering

## What is a clustering algorithm?

A **clustering algorithm** uses a set of **clustering variables** to group the data units into **clusters**, based on their similarity. Clustering methodology has been applied for a long time in many fields, under specific names. For instance, in marketing, clustering customers is called **market segmentation**. 

There are many clustering algorithms, all based on the same principle, which is to achieve the maximum similarity within clusters and the minimum similarity between clusters. This is operationalized through a **similarity measure**. We find, basically, two approaches to clustering: the **distance-based** methods, such as the $k$-means algorithm, and the **probability-based** methods, such as the EM clustering algorithm. Only the $k$-means algorithm is considered here, because most of the other methods, in spite of their popularity in the textbooks, have **scalability** problems, meaning that they do not work, or become too slow, with big data sets.

A warning note: clustering algorithms always produce clusters. But those clusters could be useless for their intended application. For instance, to contribute to a better understanding of your customers, the customer segments have to be described in an intelligible way. This would probably imply a low number of segments.

Frequently, professors and teaching materials suggest that, for clusters to be useful, their number has to be low. Though this can be adequate when the clusters are intended to be managed by humans, it may not be so in some business applications. A big e-retailer like Amazon can easily manage hundreds of clusters, with no human mind understanding them. Also, in some cases, the clusters are used only to speed up the computation. For instance, to get recommendations faster, you can search them within clusters of products.

## Similarities

In the distance-based methods, the similarity of two data units is measured by means of a distance formula, which is usually the **Euclidean distance**. The Euclidean distance is the ordinary, real-life distance. The following example illustrates this. Pick two points in a 3-dimensional space, such as $x=(2,-1,3)$ and $y=(2,2,4)$. The distance between them is

$$\textrm{dist}(x,y) = \sqrt{(2-2)^2+((-1)-2)^2+(3-4)^2}=1.414.$$

The general formula, for a $p$-dimensional space, is

$$\textrm{dist}(x,y)=\sqrt{(x_1-y_1 )^2 + \cdots + (x_p-y_p)^2}.$$

This formula can be applied to any pair of rows of a data set with $p$ numeric columns, providing a numerical similarity measure. The lower the distance between two data points, the stronger their similarity.

## Normalization

It is not rare, in real data, that some variables show a much higher variation than the rest. Formulas like the Euclidean distance make those variables too influential on the clustering process. To prevent this, the features involved in the clustering process are usually normalized. There are two favorite approaches to **normalization**, as we see next.

In **min-max normalization**, all the variables are forced, through a linear transformation, into the 0-1 range. The formula for this transformation is

$$Z=\frac{X-\min(X)}{\max(X)-\min(X)}\hbox{\thinspace}.$$

An alternative formula is **statistical standardization**, based on the mean and the standard deviation. It is frequently used in statistical analysis, since some methods require the variables involved to have zero mean and unit variance.

$$Z=\frac{X-\bar X}{\textrm{std}(X)}\hbox{\thinspace}.$$

These transformations are available in some Python packages, like scikit-learn. You can also create a specific function for your preferred normalization formula. For the min-max normalization, this would be:

```
def normalize(x): 
    return (x - x.min())/(x.max() - x.min())
```
For the statistical standardization:

```
def standardize(x):
    return (x - x.mean())/x.std()
```

These functions can be applied to any numeric column of a Pandas data frame. Mind that normalization can change significantly your clusters. In customer segmentation, for instance, this is a relevant issue.

## Cluster centers

Suppose that you wish to group the data units in $k$ clusters using $p$ numeric variables. Many clustering algorithms are based on finding a set of $k$ points, called **centers**, in the $p$-dimensional space of those variables, and clustering the units around the centers. Every unit will be assigned to the cluster whose center is most similar to that unit. The centers can also be used to assign a cluster to a new unit that has not been used to extract the centers. That cluster would be the one whose center is closer to the new unit.

In real-world applications, we look at the center as the "typical element'' of the cluster. The values that the center takes for the different variables are used to produce a description of the cluster, as far as that makes sense. This is the typical approach in customer segmentation. So a marketing manager can describe a segment of customers as individuals above 60, with annual family income between $100,000 and $250,000, who frequently watch soap opera TV series. This would be nothing but a description of the center of that segment.

The algorithms that follow this approach differ on how the centers are extracted from the data. Let us see how it works in the $k$-means clustering algorithm.

## *k*-means clustering

The **$k$-means algorithm** searches for a set of $k$ centers such that the corresponding clusters have any of the mathematically equivalent properties:

* The center is the average of the units of the cluster.

* The average squared (Euclidean) distance of a data unit to the center of its cluster, is minimum. If you regard the cluster center as a prediction, this works like a MSE loss.

The $k$-means search is iterative. The steps are:

* Take a random choice of $k$ units as the initial set of centers.

* Create $k$ clusters so that every unit belongs to the same cluster as the closest center (in the  Euclidean distance).

* Take the average of every cluster as the new center and cluster the samples around the new centers.

* Iterate until a prespecified stopping criterion is met.

* The outcome consists in a matrix containing the final cluster centers and a vector containing the cluster labels, which indicate the cluster membership.

Despite some drawbacks, $k$-means remains the most widely used clustering algorithm. It is simple, easily understandable and reasonably scalable. Moreover, it can be easily modified to deal with streaming data.

In $k$-means clustering, you have to specify the number of clusters. Even if this is something on which you do not have a definite number, you will probably have a preliminary idea, so you can work around that. For instance, you may wish to have a number of clusters from 3 to 6. So, you will try $k = 3, 4, 5, 6$, comparing the results. You will probably consider the cluster sizes, since you do not want clusters which are too small, and you will monitor how the clusters change when you increase the number of clusters.

**Stability** is expected from a respectable segmentation. Mind that, due to the random start, two runs of the $k$-means clustering can give different results. The difference should not be relevant if the clusters mean something.

## *k*-means clustering in Python

$k$-means clustering is available in the Python packages SciPy and scikit-learn, both included in the Anaconda distribution. It is not difficult to manage if you have a clear mind about what you are going to get, more specifically:

* A vector, of length equal to the number of rows of the data set, containing the **cluster labels**.

* A matrix containing the **cluster centers**, with one row for every center. For every row, the terms of that row would be the mean values of the clustering variables on the corresponding cluster.

In scikit-learn, $k$-means clustering is provided by the class `KMeans()`, from the subpackage `cluster`. Suppose that you aim at four clusters. You start as usual in scikit-learn, by instantiating your estimator. Since the default number of clusters is `n_clusters=8`, you have to change that: 

```
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=4)
```

The method `.fit()` is applied to a feature matrix (no target vector `y` here):

```
clus.fit(X)
```

There are two interesting attributes here. First, `.labels_` is a 1D array containing a cluster label for every sample. With `n_clusters=4`, the labels would be 0, 1, 2 and 3. 

```
labels = clus.labels_
```

The attribute `.cluster_centers_` is a matrix containing the coordinates of the cluster centers. In this case, this would be a 2D array with four rows, one for every center. For every row, the terms of that row would be the average feature values in the corresponding cluster.

```
centers = clus.cluster_centers_
```

Taking a look at the matrix centers can help to describe the clusters. It is recommended to normalize the feature matrix, for instance using with min-max normalization.
