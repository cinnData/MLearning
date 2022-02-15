## Example - The churn model ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
fname = path + 'churn.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:5]
round(np.mean(data['churn']), 3)

# Target vector and feature matrix #
y = data['churn']
X = data[list(data.dtype.names[1:-1])]
from numpy.lib.recfunctions import structured_to_unstructured
X = structured_to_unstructured(X)
X.shape

# Logistic regression equation #
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(max_iter=1500)
logclf.fit(X, y)
round(logclf.score(X, y), 3)

# Predictive scores #
scores = logclf.predict_proba(X)[:, 1]

# Distribution of the predictive scores #
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

# The default cutoff #
ypred = logclf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, ypred)
round(np.mean(y == ypred), 3)

# A lower cutoff #
ypred = (scores > 0.2).astype('int')
confusion_matrix(y, ypred)
round(np.mean(y == ypred), 3)
round(np.mean(ypred[y == 1]), 3)
round(np.mean(ypred[y == 0]), 3)

