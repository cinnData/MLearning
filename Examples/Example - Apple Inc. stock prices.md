# Example - Apple Inc. stock prices

## Introduction

This example is based on data on the **Apple Inc.** stock prices in the **Nasdaq** stock market, for the year 2019, as published by **Yahoo Finance**. These data are used to illustrate the way you manage data in NumPy.

## The data set

The data set (file `aapl.csv`) covers 251 **trading days**. The data come in the typical **OHLC format** (Open/High/Low/Close).

The variables are:

* `date`, the date, as `'yyyy-mm-dd'`.

* `open`, the price (US dollars) of the stock at the beginning of the trading day. It can be different from the closing price of the previous trading day.

* `high`, the highest price (US dollars) of the stock on that trading day.

* `low`, the lowest price (US dollars) of the stock on that day.

* `close`, the price (US dollars) of the stock at closing time.

* `adj_close`, the closing price adjusted for factors in corporate actions, such as stock splits, dividends, and rights offerings.

* `volume`, the amount of Apple stock that has been traded on that day.

Source: `finance.yahoo.com/quote/AAPL/history?p=AAPL`.

## Questions

Q1. Is there a trend in the opening price? 

Q2. The daily returns of an asset are derived from the asset prices as follows. If $P_t$ is the price at day $t$, the daily return at this day is given by 

$$r(t) = \frac{p(t)}{p(t−1)} − 1.$$

Calculate the daily returns for the opening price. 

Q3. How is the distribution of the daily returns? 

Q4. How is the distribution of the trading volume? 

Q5. Is there an association between the daily returns and the trading volume?
