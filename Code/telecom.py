## Example - Telecom churn prediction ##

# Importing the data #
import csv
import numpy as np
with open('Dropbox/ml_course/data/telecom.csv', mode='r') as conn:
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
y = (Xy[:, 20] == 'Yes').astype(int)
y.shape
list1 = [2, 5, 18, 19]
list2 = [1] + list(range(3, 5)) + list(range(6, 18))
X1 = Xy[:, np.array(list1)].astype(float)
X2 = Xy[:, np.array(list2)]
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(X2)
X2 = enc.transform(X2).toarray()
X2.shape
X = np.concatenate([X1, X2], axis=1)

# Churning rate #
y.mean().round(3)

# Q1. Logistic regression model #
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
scores = clf.predict_proba(X)[:, 1]

# Q2. Distribution of churn scores #
from matplotlib import pyplot as plt
plt.figure(figsize = (14,6))
plt.subplot(1, 2, 1)
plt.hist(scores[y == 1], color='gray', rwidth=0.95, bins=17, range=(0,0.85))
plt.title('Figure a. Scores (Churners)')
plt.xlabel('Churn score')
plt.subplot(1, 2, 2)
plt.hist(scores[y == 0], color='gray', rwidth=0.95, bins=17, range=(0,0.85))
plt.title('Figure b. Scores (non-churners)')
plt.xlabel('Churn score');

# Q3. Cutoff 0.3 #
y_pred = (scores > 0.3).astype(int)
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y, y_pred)
conf
tp = conf[1, 1]/sum(conf[1, :])
fp = conf[0, 1]/sum(conf[0, :])
round(tp, 3), round(fp, 3)

# Q4a. Train/test split #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
score_train = clf.predict_proba(X_train)[:, 1]
conf_train = confusion_matrix(y_train, score_train > 0.3)
tp_train = conf_train[1, 1]/sum(conf_train[1, :])
fp_train = conf_train[0, 1]/sum(conf_train[0, :])
round(tp_train, 3), round(fp_train, 3)
score_test = clf.predict_proba(X_test)[:, 1]
conf_test = confusion_matrix(y_test, score_test > 0.3)
tp_test = conf_test[1, 1]/sum(conf_test[1, :])
fp_test = conf_test[0, 1]/sum(conf_test[0, :])
round(tp_test, 3), round(fp_test, 3)

# Q4b. Cross-validation #
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_train, y_train, cv=3).round(3)
clf.score(X_test, y_test).round(3)
