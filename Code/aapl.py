## Example - Apple Inc. stock prices ##

# Importing the data (edit path) #
import csv
with open('Dropbox/ml_course/data/aapl.csv', 'r') as conn:
	reader = csv.reader(conn)
	data = list(reader)
len(data)
data[0]
data[1]

# Organizing the data #
header = data[0]
header
import numpy as np
X = np.array(data[1:])[:, 1:].astype(float)
X[:5, :]

# Q1. Trend in the opening price #
price = X[:, 0]
price.shape
price[:5]
from matplotlib import pyplot as plt
plt.figure(figsize=(10,6))
plt.title('Figure 1. Opening price')
plt.plot(price, color='gray')
plt.ylabel('US dollar');
t = np.arange(251)
t[:10]
np.corrcoef(t, price)
np.corrcoef(t, price)[0, 1].round(3)

# Q2. Daily returns #
returns = 100*(price[1:]/price[:-1] - 1)
plt.figure(figsize=(10,6))
plt.title('Figure 2. Daily returns')
plt.plot(returns, color='gray')
plt.ylabel('Percent return');

# Q3. Distribution of the daily returns #
plt.figure(figsize=(8,6))
plt.title('Figure 3. Daily returns')
plt.hist(returns, color='gray', rwidth=0.97)
plt.xlabel('Percent return')
plt.ylabel('Frequency');

# Q4. Distribution of the trading volume #
volume = X[:, 5]/10**6
volume[:5]
plt.figure(figsize=(8,6))
plt.title('Figure 4. Trading volume')
plt.hist(volume, color='gray', rwidth=0.97)
plt.xlabel('Million shares')
plt.ylabel('Frequency');

# Q5. Association between daily returns and trading volume #
plt.figure(figsize=(6,6))
plt.title('Figure 5. Trading volume vs returns')
plt.scatter(x=volume[1:], y=returns, color='gray', s=25)
plt.xlabel('Trading volume')
plt.ylabel('Daily returns');
np.corrcoef(volume[1:], returns)[0, 1].round(3)
np.corrcoef(volume[1:], np.abs(returns))[0, 1].round(3)
