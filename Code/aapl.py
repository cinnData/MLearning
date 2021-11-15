## Example - Apple Inc. stock prices ##

# Importing the data #
import numpy as np
path = 'https://raw.githubusercontent.com/cinnData/MLearning/main/Data/'
fname = path + 'aapl.csv'
data = np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')
data.shape
data[:5]

# Is there a trend in the opening price? #
open = data['open']
type(open)
open.shape
from matplotlib import pyplot as plt
plt.figure(figsize=(10,6))
plt.title('Figure 1. Opening price')
plt.plot(open, color='gray');
t = np.arange(251)
t[:10]
np.corrcoef(t, open)
np.corrcoef(t, open)[0, 1].round(3)

# Daily returns #
returns = 100*(open[1:]/open[:-1] - 1)
plt.figure(figsize=(10,6))
plt.title('Figure 2. Daily returns')
plt.plot(returns, color='gray');
plt.figure(figsize=(8,6))
plt.title('Figure 3. Daily returns')
plt.hist(returns, color='gray', rwidth=0.97);

# Distribution of the trading volume #
volume = data['volume']
plt.figure(figsize=(8,6))
plt.title('Figure 4. Trading volume')
plt.hist(volume, color='gray', rwidth=0.97);

# Association between daily returns and trading volume #
plt.figure(figsize=(6,6))
plt.title('Figure 5. Trading volume vs returns')
plt.scatter(x=volume[1:], y=returns, color='gray', s=25)
plt.xlabel('Trading volume')
plt.ylabel('Daily returns');
np.corrcoef(volume[1:], returns)[0, 1].round(3)
np.corrcoef(volume[1:], np.abs(returns))[0, 1].round(3)
