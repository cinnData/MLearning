## Example - The churn model ##

# Importing the data #
import csv
with open('Dropboxml_course/data/churn.csv', mode='r') as conn:
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
y = Xy[:, 13].astype(float)
y.shape
X = Xy[:, 1:13].astype(float)
X.shape

# Churning rate #
y.mean().round(3)

# Q1a. Logistic regression model #
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1500)
clf.fit(X, y)
round(clf.score(X, y), 3)

# Q1b. Predictive scores #
scores = clf.predict_proba(X)[:, 1]

# Q2. Distribution of the predictive scores #
from matplotlib import pyplot as plt
# Set the size of the figure
plt.figure(figsize = (14,6))
# First subplot
plt.subplot(1, 2, 1)
plt.hist(scores[y == 1], range=(0,1), color='gray', rwidth=0.96)
plt.title('Figure a. Scores (Churners)')
plt.xlabel('Churn score')
# Second subplot
plt.subplot(1, 2, 2)
plt.hist(scores[y == 0], range=(0,1), color='gray', rwidth=0.96)
plt.title('Figure b. Scores (non-churners)')
plt.xlabel('Churn score');

# Q3a. The default cutoff #
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)
round(np.mean(y == y_pred), 3)

# Q3b. A lower cutoff #
y_pred = (scores > 0.2).astype('int')
confusion_matrix(y, y_pred)
round(np.mean(y == y_pred), 3)
round(np.mean(y_pred[y == 1]), 3)
round(np.mean(y_pred[y == 0]), 3)
