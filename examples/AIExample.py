
# coding: utf-8

# # Strategy Analysis with **Pandas TA** and AI/ML
# * This is a **Work in Progress** and subject to change!
# * Contributions are welcome and accepted!
# * Examples below are for **educational purposes only**

# ### Required Packages
# ##### Uncomment the packages you need to install or are missing

# In[1]:


#!pip install numpy 1.18.3
#!pip install pandas 1.1.0
#!pip install mplfinance 0.12.6a3 in matplotlib
#!pip install pandas-datareader
#!pip install requests_cache
#!pip install alphaVantage-api
# Pandas TA 0.2.02

# In[1]:


# get_ipython().run_line_magic('pylab', 'inline')
import os, pdb, sys
import datetime as dt
import random as rnd
from sys import float_info as sflt

import numpy as np
import pandas as pd
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 20)

# Error spot
import mplfinance as mpf
from alphaVantageAPI.alphavantage import AlphaVantage

import pandas_ta as ta

from watchlist import colors, Watchlist

print(f"Numpy v{np.__version__}")
print(f"Pandas v{pd.__version__}")
print(f"mplfinance v{mpf.__version__}")
print(f"Pandas TA v{ta.version}")
# get_ipython().run_line_magic('matplotlib', 'inline')

DEBUGGING = True

# ## MISC Functions

# In[6]:


def recent_bars(df, tf: str = "1y"):
    # All Data: 0, Last Four Years: 0.25, Last Two Years: 0.5, This Year: 1, Last Half Year: 2, Last Quarter: 3
    yearly_divisor = {"all": 0, "10y": 0.1, "5y": 0.2, "4y": 0.25, "3y": 1./3, "2y": 0.5, "1y": 1, "6mo": 2, "3mo": 3}
    yd = yearly_divisor[tf] if tf in yearly_divisor.keys() else 0
    return int(ta.TRADING_DAYS_PER_YEAR / yd) if yd > 0 else df.shape[0]


# ## Collect some Data

# In[7]:


tf = "D"
tickers = ["SPY", "QQQ", "AAPL", "TSLA"]
watch = Watchlist(tickers, tf=tf)
watch.strategy = ta.CommonStrategy
watch.load(tickers, timed=True, analyze=True, verbose=False)


# # Select an Asset

# In[ ]:


ticker = tickers[2]
# watch.data[ticker].ta.constants(True, [0, 0, 0])
print(f"{ticker} {watch.data[ticker].shape}\nColumns: {', '.join(list(watch.data[ticker].columns))}")


# ### Trim it

# In[ ]:


duration = "1y"
recent = recent_bars(watch.data[ticker], duration)
asset = watch.data[ticker].copy().tail(recent)
print(f"{ticker} {asset.shape}\nColumns: {', '.join(list(asset.columns))}")


# # Create a Trend

# In[ ]:


# Example Long Trends
# long = ta.sma(asset.close, 10) < ta.sma(asset.close, 20) # SMA(10) > SMA(20)
# long = ta.ema(asset.close, 8) > ta.ema(asset.close, 21) # EMA(8) > EMA(21)
long = ta.increasing(ta.ema(asset.close, 50))
# long = ta.macd(asset.close).iloc[:,1] > 0 # MACD Histogram is positive

asset.ta.ema(length=8, append=True)
asset.ta.ema(length=21, append=True)
asset.ta.ema(length=50, append=True)


# ## Calculate Trend Returns from the long trend

# In[ ]:


trendy = asset.ta.trend_return(trend=long, cumulative=True, append=True)
trendy.tail() # Third Column is the long trend; binary sequences


# ## Trend Returns and Cumulative Trend Returns

# In[ ]:


cltr = trendy.iloc[:,0]
tr = trendy.iloc[:,1]

trendy.iloc[:,:2].plot(figsize=(16, 3), color=colors("BkBl"))
cltr.plot(figsize=(16, 3), kind="area", stacked=False, color=colors("SvGy")[0], alpha=0.25, grid=True)


# # Total Return

# In[ ]:


capital = 10000

total_return = cltr.cumsum() * capital
positive_return = total_return[total_return > 0]
negative_return = total_return[total_return <= 0]
trdf = pd.DataFrame({"tr+": positive_return, "tr-": negative_return})
trdf.plot(figsize=(16, 5), color=colors(), kind="area", stacked=False, alpha=0.25, grid=True)


# ## Long and Short Trends

# In[ ]:

# date
# 2019-09-25    0
# 2019-09-26    0
# 2019-09-27    0
# 2019-09-30    0
long_trend = (trendy.iloc[:,-2] > 0).astype(int)
short_trend = (1 - long_trend).astype(int)

# Gap checking
# date
# 2019-09-25      NaT
# 2019-09-26   1 days
# 2019-09-27   1 days
# 2019-09-30   3 days
mask = long_trend.index.to_series().diff()  # <class 'pandas.core.series.Series'>


long_trend.plot(figsize=(16, 0.85), kind="area", stacked=True, color=colors()[0], alpha=0.25)
short_trend.plot(figsize=(16, 0.85), kind="area", stacked=True, color=colors()[1], alpha=0.25)

if (DEBUGGING):
    pdb.set_trace()

# ## Entries & Exits

# In[ ]:


entries = (trendy.iloc[:,-1] > 0).astype(int) * asset.close
entries[entries < 0.0001] = np.NaN
entries.name = "entries"
exits = (trendy.iloc[:,-1] < 0).astype(int) * asset.close
exits[exits < 0.0001] = np.NaN
exits.name = "exits"

first_date = asset.index[0]
last_date = asset.index[-1]
f_date = f"{first_date.day_name()} {first_date.month}-{first_date.day}-{first_date.year}"
l_date = f"{last_date.day_name()} {last_date.month}-{last_date.day}-{last_date.year}"
last_ohlcv = f"Last OHLCV: ({asset.iloc[-1].open}, {asset.iloc[-1].high}, {asset.iloc[-1].low}, {asset.iloc[-1].close}, {int(asset.iloc[-1].volume)})"
ptitle = f"\n{ticker} ({tf} | {duration}) from {f_date} to {l_date}  ({recent} bars)\n{last_ohlcv}"

# chart = asset["close"] #asset[["close", "SMA_10", "SMA_20", "SMA_50", "SMA_200"]]
# chart = asset[["close", "SMA_10", "SMA_20"]]
chart = asset[["close", "EMA_8", "EMA_21", "EMA_50"]]
chart.plot(figsize=(16, 10), color=colors("BkGrRd"), title=ptitle, grid=True)
entries.plot(figsize=(16, 10), color=colors("FcLi")[1], marker="^", markersize=12, alpha=0.8)
exits.plot(figsize=(16, 10), color=colors("FcLi")[0], marker="v", markersize=12, alpha=0.8, grid=True)

total_trades = trendy.iloc[:,-1].abs().sum()
print(f"Total Trades: {total_trades}")

entries_ = entries.dropna()
exits_ = exits.dropna()
all_trades = trendy.iloc[:,-1].copy().dropna()
# all_trades[all_trades != 0]

if (DEBUGGING):
    pdb.set_trace()

# # AI Analysis

# binary sequence ---> software ---> entries and exits

# In[ ]:


import pdb
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[ ]:


def make_dataset(n_chars, n_vocab, raw_text, char_to_int):
	# prepare the dataset of input to output pairs encoded as integers

	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print ("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	if (TESTING):
		pdb.set_trace()

	return X, y


# In[ ]:


def train_model(X, y):
	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# define the checkpoint
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	# fit the model
	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


# In[ ]:


file_name = input("Please, give input text file, thanks.")
n_chars, n_vocab, raw_text, char_to_int = data_manipulation(file_name)
X, y = make_dataset(n_chars, n_vocab, raw_text, char_to_int)
train_model(X, y)
