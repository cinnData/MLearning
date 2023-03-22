## Example - Marketing to frequent fliers ##

# Importing the data #
import csv
import numpy as np
with open('Dropbox/ML-2023-Winter/data/fliers.csv', mode='r') as conn: 
  reader = csv.reader(conn)
  data = list(reader)
len(data)

# Headers #
header = data[0]
header
len(header)

# Feature matrix #
import numpy as np
X = np.array(data[1:])[:, 1:].astype(float)
X.shape

# Exploratory analysis #
from matplotlib import pyplot as plt
plt.figure(figsize = (14,12))
plt.subplot(2, 2, 1)
plt.title('Figure a. Miles eligible for travel award')
plt.hist(X[:, 0]/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand miles')
plt.subplot(2, 2, 2)
plt.title('Figure b. Miles qualifying for Topflight status')
plt.hist(X[:, 1]/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand miles')
plt.subplot(2, 2, 3)
plt.title('Figure c. Miles from non-flight bonus transactions')
plt.hist(X[:, 5]/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand miles')
plt.subplot(2, 2, 4)
plt.title('Figure d. Days since the customer was enrolled')
plt.hist(X[:, 9]/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand days');
np.unique(X[:, 2], return_counts=True)
np.unique(X[:, 3], return_counts=True)
np.unique(X[:, 4], return_counts=True)

# Q1. 4-cluster analysis #
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=4, random_state=0)
clus.fit(X)
labels = clus.labels_
centers = clus.cluster_centers_
np.unique(labels, return_counts=True)
centers.shape
import pandas as pd
pdcenters = pd.DataFrame(centers).round(1)
pdcenters.columns = header[1:]
pdcenters['size'] = np.unique(labels, return_counts=True)[1]
pdcenters

# Q2a. Normalization #
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
Z = scaler.transform(X)

# Q2b. 4-cluster analysis, after normalization #
clus.fit(Z)
labels = clus.labels_
centers = clus.cluster_centers_
pdcenters = pd.DataFrame(centers).round(3)
pdcenters.columns = header[1:]
pdcenters['size'] = np.unique(labels, return_counts=True)[1]
pdcenters
