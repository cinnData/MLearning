## [MLE-03] Example - The churn model ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'churn.csv', index_col=0)

# Exploring the data #
df.info()
df.index.duplicated().sum() + df.duplicated().sum()

# Q1. Logistic regression equation #
y = df['churn']
X = df.drop(columns='churn')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1500)
clf.fit(X, y)
clf.score(X, y).round(3)
clf.predict_proba(X)
df['score'] = clf.predict_proba(X)[:, 1]
df[['churn', 'score']]

# Q2. Distribution of the predictive scores #
from matplotlib import pyplot as plt
# Set the size of the figure
plt.figure(figsize = (12,5))
# First subplot
plt.subplot(1, 2, 1)
plt.hist(df['score'][y == 1], range=(0,1), color='gray', edgecolor='white')
plt.title('Figure 1.a. Scores (churners)')
plt.xlabel('Churn score')
# Second subplot
plt.subplot(1, 2, 2)
plt.hist(df['score'][y == 0], range=(0,1), color='gray', edgecolor='white')
plt.title('Figure 1.b. Scores (non-churners)')
plt.xlabel('Churn score');

# Q3. Set a threshold for the churn scores #
y_pred = clf.predict(X)
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)
(y == y_pred).mean().round(3)
y_pred = (df['score'] > 0.2).astype(int)
confusion_matrix(y, y_pred)
(y == y_pred).mean().round(3)
y_pred[y == 1].mean().round(3)
y_pred[y == 0].mean().round(3)
