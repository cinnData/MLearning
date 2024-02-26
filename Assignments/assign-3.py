## [MLA-03] Assignment 3 ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'deposit.csv', index_col=0)
y = df['deposit']
X = df.drop(columns='deposit')

# Q1a. Train-test split #
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.5, random_state=0)
df_train.shape, df_test.shape
df_train['deposit'].sum(), df_test['deposit'].sum()

# Q1b. Scoring #
y_train, X_train = df_train['deposit'], df_train.drop(columns='deposit')
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(max_iter=2000)
clf1.fit(X_train, y_train)
y_test, X_test = df_test['deposit'], df_test.drop(columns='deposit')
df_test['score'] = clf1.predict_proba(X_test)[:, 1]

# Q1c. Evaluating the model in the test set #
df_test.sort_values('score', inplace=True, ascending=False)
N = int(len(df_test)/5)
N
conv_rate = df_test['deposit'].head(N).sum()/N
round(conv_rate, 3)

# Q2a. Undersampling #
df_train_pos = df_train[df_train['deposit'] == 1]
df_train_neg = df_train[df_train['deposit'] == 0]
df_train_neg = df_train_neg.sample(n=len(df_train_pos), replace=False)
df_train_under = pd.concat([df_train_pos, df_train_neg])
df_train_under.shape
df_test = df_test.drop(columns='score')

# Q2b. Logistic regression classifier #
y_train, X_train = df_train_under['deposit'], df_train_under.drop(columns='deposit')
y_test, X_test = df_test['deposit'], df_test.drop(columns='deposit')
clf2 = LogisticRegression(max_iter=2000)
clf2.fit(X_train, y_train)

# Q2c. Testing #
y_pred = clf2.predict(X_test)
conf = pd.crosstab(y_test, y_pred)
conf
acc = (y_test == y_pred).mean().round(3)
prec = y_test[y_pred == 1].mean().round(3)
rec = y_pred[y_test == 1].mean().round(3)
acc, prec, rec

# Q3a. Oversampling #
df_train_pos = df_train[df_train['deposit'] == 1]
df_train_neg = df_train[df_train['deposit'] == 0]
df_train_pos = df_train_pos.sample(n=len(df_train_neg), replace=True)
df_train_over = pd.concat([df_train_pos, df_train_neg])
df_train_over.shape

# Q3b. Logistic regression classifier #
y_train, X_train = df_train_over['deposit'], df_train_over.drop(columns='deposit')
clf3 = LogisticRegression(max_iter=2000)
clf3.fit(X_train, y_train)

# Q3c. Testing #
y_pred = clf3.predict(X_test)
conf = pd.crosstab(y_test, y_pred)
conf
acc = (y_test == y_pred).mean().round(3)
prec = y_test[y_pred == 1].mean().round(3)
rec = y_pred[y_test == 1].mean().round(3)
acc, prec, rec
