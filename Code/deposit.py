## [MLE-04]  Example - Direct marketing of term deposits ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'deposit.csv', index_col=0)
df.info()
df.head()
df['deposit'].mean().round(3)

# Q1. Logistic regression model #
y = df['deposit']
X = df.drop(columns='deposit')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000)
clf.fit(X, y)
y_pred = clf.predict(X)
conf = pd.crosstab(y, y_pred)
conf
acc = (y == y_pred).mean().round(3)
acc1 = y_pred[y == 1].mean().round(3)
acc0 = (1 - y_pred[y == 0]).mean().round(3)
acc, acc1, acc0

# Q2. Predictive scores #
df['score'] = clf.predict_proba(X)[:, 1]
df[['deposit', 'score']]
df['score'].describe()
from matplotlib import pyplot as plt
# Set the size of the figure
plt.figure(figsize=(12,5))
# First subplot
plt.subplot(1, 2, 1)
plt.hist(df['score'][y == 1], color='gray', edgecolor='white')
plt.title('Figure 1.a. Scores (subscribers)')
plt.xlabel('Subscription score')
# Second subplot
plt.subplot(1, 2, 2)
plt.hist(df['score'][y == 0], color='gray', edgecolor='white')
plt.title('Figure 1.b. Scores (non-subscribers)')
plt.xlabel('Subscription score');

# Q3. Set a threshold for the scores #
y_pred = (df['score'] > 0.11).astype(int)
conf = pd.crosstab(y, y_pred)
conf
acc = (y == y_pred).mean().round(3)
acc1 = y_pred[y == 1].mean().round(3)
acc0 = (1 - y_pred[y == 0]).mean().round(3)
acc, acc1, acc0

# Q4. Target of 4,000 subscriptions
df.sort_values('score', inplace=True, ascending=False)
df[['deposit', 'score']]
df['cum_subscription'] = df['deposit'].cumsum()
df[['deposit', 'score', 'cum_subscription']]
(df['cum_subscription'] < 4000).sum() + 1

# Q5. Budget 10,000 calls #
call_list = df.index[:10000]
call_list
df['cum_subscription'][call_list[9999]]
