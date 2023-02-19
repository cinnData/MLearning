## Example - The spam filter ##

# Importing the data #
import csv
import numpy as np
with open('Dropbox/ml_course/data/spam.csv', mode='r') as conn:
	reader = csv.reader(conn)
	data = list(reader)
len(data)

# Headers #
header = data[0]
header
len(header)

# Target vector and feature matrix #
import numpy as np
Xy = np.array(data[1:])
Xy.shape
y = Xy[:, 51].astype(float)
y.shape
X = Xy[:, :51].astype(float)
X.shape

# Spam rate #
y.mean().round(3)

# Q1a1. Decision tree classifier (max depth = 2) #
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(max_depth=2)
clf1.fit(X, y)
round(clf1.score(X, y), 3)

# Q1a2. Confusion matrix #
y_pred1 = clf1.predict(X)
from sklearn.metrics import confusion_matrix
conf1 = confusion_matrix(y, y_pred1)
conf1
tp1 = conf1[1, 1]/np.sum(conf1[1, :])
fp1 = conf1[0, 1]/np.sum(conf1[0, :])
round(tp1, 3), round(fp1, 3)

# Q1b. Decision tree classifier (max depth = 3) #
clf2 = DecisionTreeClassifier(max_depth=3)
clf2.fit(X, y)
y_pred2 = clf2.predict(X)
conf2 = confusion_matrix(y, y_pred2)
conf2
tp2 = conf2[1, 1]/np.sum(conf2[1, :])
fp2 = conf2[0, 1]/np.sum(conf2[0, :])
round(tp2, 3), round(fp2, 3)

# Q1c. Decision tree classifier (max depth = 4) #
clf3 = DecisionTreeClassifier(max_depth=4)
clf3.fit(X, y)
y_pred3 = clf3.predict(X)
conf3 = confusion_matrix(y, y_pred3)
conf3
tp3 = conf3[1, 1]/np.sum(conf3[1, :])
fp3 = conf3[0, 1]/np.sum(conf3[0, :])
round(tp3, 3), round(fp3, 3)

# Q1d. Decision tree classifier (max depth = 5) #
clf4 = DecisionTreeClassifier(max_depth=5)
clf4.fit(X, y)
y_pred4 = clf4.predict(X)
conf4 = confusion_matrix(y, y_pred4)
conf4
tp4 = conf4[1, 1]/np.sum(conf4[1, :])
fp4 = conf4[0, 1]/np.sum(conf4[0, :])
round(tp4, 3), round(fp4, 3)

# Q2. Feature relevance #
imp = clf4.feature_importances_
feat_list = np.array(header[:-1])[imp > 0]
feat_imp = np.round(clf4.feature_importances_[imp > 0], 3)
feat_report = np.array([feat_list, feat_imp], dtype=object).transpose()
feat_report[np.argsort(np.array(feat_report[:, 1], dtype='float'))[::-1], :]
