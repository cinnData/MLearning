## [MLA-02] Assignment 2 ##

# Importing the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'spam.csv')

# Target vector and feature matrix #
y = df['spam']
X = df.drop(columns='spam')

# Evaluation #
from sklearn.metrics import confusion_matrix
def eval1(clf):
        y_pred = clf.predict(X)
        conf = confusion_matrix(y, y_pred)
        tp = conf[1, 1]/sum(conf[1, :])
        fp = conf[0, 1]/sum(conf[0, :])
        return round(tp, 3), round(fp, 3)

# Q1a. Logistic regression classifier #
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(max_iter=2000)
clf1.fit(X, y)
eval1(clf1)

# Q1b. Decision tree classifier #
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier(max_depth=5)
clf2.fit(X, y)
eval1(clf2)
clf2.feature_importances_

# Q2a. Binary data set #
BX = (X.iloc[:, :-3] > 0).astype('int')
BX.head()
def eval2(clf):
        y_pred = clf.predict(BX)
        conf = confusion_matrix(y, y_pred)
        tp = conf[1, 1]/sum(conf[1, :])
        fp = conf[0, 1]/sum(conf[0, :])
        return round(tp, 3), round(fp, 3)

# Q2b. Logistic regression classifier #
clf3 = LogisticRegression()
clf3.fit(BX, y)
eval2(clf3)

# Q2c. Decision tree classifier #
clf4 = DecisionTreeClassifier(max_depth=5)
clf4.fit(BX, y)
eval2(clf4)
clf4.feature_importances_
