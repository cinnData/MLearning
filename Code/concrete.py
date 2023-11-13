## [MLE-01] Example - Modeling the strength of concrete ##

# Q1. Import the data #
import pandas as pd
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
df = pd.read_csv(path + 'concrete.csv')
df.info()
df.head()

# Q2. Target vector and feature matrix #
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Q3. Linear regression model #
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)

# Q4. Predicted strength values #
y_pred = reg.predict(X)
X_new = df.describe().iloc[1:2, :-1].round()
X_new
reg.predict(X_new)

# Q5. Evaluate the model #
reg.score(X, y).round(3)
from matplotlib import pyplot as plt
plt.figure(figsize=(5,5))
plt.title('Figure 1. Actual strength vs predicted strength')
plt.scatter(y_pred, y, color='black', s=2)
plt.xlabel('Predicted strength')
plt.ylabel('Actual strength');

# Q6. Save the model for future use #
import joblib
joblib.dump(reg, 'reg.pkl')
newreg = joblib.load('reg.pkl')
(reg.predict(X) != newreg.predict(X)).sum()
