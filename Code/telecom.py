## Example - Telecom churn prediction ##

# Importing data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
fname = path + 'telecom.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:3]
round(np.mean(data['churn'] == 'Yes'), 3)

# Target vector and feature matrix #
y = (data['churn'] == 'Yes').astype('int')
X1 = data[['senior_citizen', 'tenure', 'monthly_charges', 'total_charges']]
from numpy.lib.recfunctions import structured_to_unstructured
X1 = structured_to_unstructured(X1)
X2 = data[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'internet_service',
  'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
  'streaming_movies', 'contract', 'paperless_billing', 'payment_method']]
X2 = structured_to_unstructured(X2)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(X2)
X2 = enc.transform(X2).toarray()
X2.shape
np.unique(X2, return_counts=True)
X = np.concatenate([X1, X2], axis=1)

# A logistic regression classifier #
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(max_iter=500)
logclf.fit(X, y)
round(logclf.score(X, y), 3)
scores = logclf.predict_proba(X)[:, 1]
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
ypred = (scores > 0.3).astype('int')
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y, ypred)
conf
tp = conf[1, 1]/sum(conf[1, :])
fp = conf[0, 1]/sum(conf[0, :])
round(tp, 3), round(fp, 3)

# Splitting in training and test data sets #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
logclf.fit(X_train, y_train)
score_train = logclf.predict_proba(X_train)[:, 1]
conf_train = confusion_matrix(y_train, score_train > 0.3)
tp_train = conf_train[1, 1]/sum(conf_train[1, :])
fp_train = conf_train[0, 1]/sum(conf_train[0, :])
round(tp_train, 3), round(fp_train, 3)
score_test = logclf.predict_proba(X_test)[:, 1]
conf_test = confusion_matrix(y_test, score_test > 0.3)
tp_test = conf_test[1, 1]/sum(conf_test[1, :])
fp_test = conf_test[0, 1]/sum(conf_test[0, :])
round(tp_test, 3), round(fp_test, 3)
