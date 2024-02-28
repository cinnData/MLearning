## [MLE-10] Clustering examples ##

# Importing the spam data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'spam.csv')
y = df['spam']
X = df.drop(columns='spam')

# Q1. 2-cluster analysis (original data) #
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=2, random_state=0)
clus.fit(X)
label1 = clus.labels_
label1
pd.crosstab(y, label1)
(y == label1).mean().round(3)

# Q2. Binary data set #
BX = (X.iloc[:, :-3] > 0).astype('int')
clus.fit(BX)
label2 = clus.labels_
pd.crosstab(y, label2)
(y == 1 - label2).mean().round(3)

# Q3. Removing features #
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(BX, y)
clf.feature_importances_
DBX = BX.iloc[:, clf.feature_importances_ > 0]
clus.fit(DBX)
label3 = clus.labels_
pd.crosstab(y, label3)
(y == 1 - label3).mean().round(3)
