## Example - Marketing to frequent fliers ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
fname = path + 'fliers.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:10]

# Exploratory analysis #
from matplotlib import pyplot as plt
plt.figure(figsize = (14,12))
plt.subplot(2, 2, 1)
plt.title('Figure a. Miles eligible for travel award')
plt.hist(data['balance']/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand miles')
plt.subplot(2, 2, 2)
plt.title('Figure b. Miles qualifying for Topflight status')
plt.hist(data['qual_miles']/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand miles')
plt.subplot(2, 2, 3)
plt.title('Figure c. Miles from non-flight bonus transactions')
plt.hist(data['bonus_miles']/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand miles')
plt.subplot(2, 2, 4)
plt.title('Figure d. Days since the customer was enrolled')
plt.hist(data['days_since_enroll']/1000, color='gray', rwidth=0.97)
plt.xlabel('Thousand days');
np.unique(data['cc1_miles'], return_counts=True)
np.unique(data['cc2_miles'], return_counts=True)
np.unique(data['cc3_miles'], return_counts=True)

# Feature matrix #
from numpy.lib.recfunctions import structured_to_unstructured
X = structured_to_unstructured(data)[:, 1:]

# 4-cluster analysis, first round #
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=4, random_state=0)
clus.fit(X)
labels = clus.labels_
centers = clus.cluster_centers_
np.unique(labels, return_counts=True)
centers.shape
import pandas as pd
pdcenters = pd.DataFrame(centers).round(1)
pdcenters.columns = data.dtype.names[1:]
pdcenters['size'] = np.unique(labels, return_counts=True)[1]
pdcenters

# Normalization #
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
Z = scaler.transform(X)

# 4-cluster analysis, second round #
clus.fit(Z)
labels = clus.labels_
centers = clus.cluster_centers_
pdcenters = pd.DataFrame(centers).round(3)
pdcenters.columns = data.dtype.names[1:]
pdcenters['size'] = np.unique(labels, return_counts=True)[1]
pdcenters
