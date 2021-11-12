## Example - The spam filter ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/7.%20Decision%20trees/'
fname = path + 'spam.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:1]
round(np.mean(data['spam']), 3)

# Target vector and feature matrix #
y = data['spam']
from numpy.lib.recfunctions import structured_to_unstructured
X = data[list(data.dtype.names)[:-1]]
X = structured_to_unstructured(X)
X.shape

# Decision tree classifier #
from sklearn.tree import DecisionTreeClassifier
treeclf1 = DecisionTreeClassifier(max_depth=2)
treeclf1.fit(X, y)
round(treeclf1.score(X, y), 3)

# Confusion matrix #
ypred1 = treeclf1.predict(X)
from sklearn.metrics import confusion_matrix
conf1 = confusion_matrix(y, ypred1)
conf1
tp1 = conf1[1, 1]/np.sum(conf1[1, :])
fp1 = conf1[0, 1]/np.sum(conf1[0, :])
round(tp1, 3), round(fp1, 3)

# Deeper tree #
treeclf2 = DecisionTreeClassifier(max_depth=3)
treeclf2.fit(X, y)
ypred2 = treeclf2.predict(X)
conf2 = confusion_matrix(y, ypred2)
conf2
tp2 = conf2[1, 1]/np.sum(conf2[1, :])
fp2 = conf2[0, 1]/np.sum(conf2[0, :])
round(tp2, 3), round(fp2, 3)

# Deeper tree #
treeclf3 = DecisionTreeClassifier(max_depth=4)
treeclf3.fit(X, y)
ypred3 = treeclf3.predict(X)
conf3 = confusion_matrix(y, ypred3)
conf3
tp3 = conf3[1, 1]/np.sum(conf3[1, :])
fp3 = conf3[0, 1]/np.sum(conf3[0, :])
round(tp3, 3), round(fp3, 3)

# Deeper tree #
treeclf4 = DecisionTreeClassifier(max_depth=5)
treeclf4.fit(X, y)
ypred4 = treeclf4.predict(X)
conf4 = confusion_matrix(y, ypred4)
conf4
tp4 = conf4[1, 1]/np.sum(conf4[1, :])
fp4 = conf4[0, 1]/np.sum(conf4[0, :])
round(tp4, 3), round(fp4, 3)

# Feature importance #
imp = treeclf4.feature_importances_
feat_list = np.array(data.dtype.names[:-1])[imp > 0]
feat_imp = np.round(treeclf4.feature_importances_[imp > 0], 3)
feat_report = np.array([feat_list, feat_imp], dtype=object).transpose()
feat_report[np.argsort(np.array(feat_report[:, 1], dtype='float'))[::-1], :]
