## [ML-06E] Example - The spam filter ##

# Importing the data #
import pandas as pd, numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'spam.csv')

# Exploring the data #
df.shape
df.head()
df['spam'].mean().round(3)

# Target vector and feature matrix #
y = df['spam']
X = df.drop(columns='spam')

# Q1. Decision tree classifier (max depth = 2) #
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(max_depth=2)
clf1.fit(X, y)
clf1.score(X, y).round(3)
y_pred1 = clf1.predict(X)
from sklearn.metrics import confusion_matrix
conf1 = confusion_matrix(y, y_pred1)
conf1
tp1 = conf1[1, 1]/sum(conf1[1, :])
fp1 = conf1[0, 1]/sum(conf1[0, :])
round(tp1, 3), round(fp1, 3)

# Q2. Decision tree classifier (max depth = 3) #
clf2 = DecisionTreeClassifier(max_depth=3)
clf2.fit(X, y)
y_pred2 = clf2.predict(X)
conf2 = confusion_matrix(y, y_pred2)
conf2
tp2 = conf2[1, 1]/sum(conf2[1, :])
fp2 = conf2[0, 1]/sum(conf2[0, :])
round(tp2, 3), round(fp2, 3)

# Q3. Decision tree classifier (max depth = 4) #
clf3 = DecisionTreeClassifier(max_depth=4)
clf3.fit(X, y)
y_pred3 = clf3.predict(X)
conf3 = confusion_matrix(y, y_pred3)
conf3
tp3 = conf3[1, 1]/sum(conf3[1, :])
fp3 = conf3[0, 1]/sum(conf3[0, :])
round(tp3, 3), round(fp3, 3)

# Q4. Decision tree classifier (max depth = 5) #
clf4 = DecisionTreeClassifier(max_depth=5)
clf4.fit(X, y)
y_pred4 = clf4.predict(X)
conf4 = confusion_matrix(y, y_pred4)
conf4
tp4 = conf4[1, 1]/sum(conf4[1, :])
fp4 = conf4[0, 1]/sum(conf4[0, :])
round(tp4, 3), round(fp4, 3)

# Q5. Feature relevance #
imp = clf4.feature_importances_
imp
feat_list = pd.Series(imp, index=df.columns[:51])
feat_list[imp > 0].sort_values(ascending=False).round(3)
