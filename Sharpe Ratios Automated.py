#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime as dt
import math
import numpy as np


# In[11]:


lookback = 252

stocks = "GOOG AMZN MSFT AAPL".split()
start = dt.date.today() - dt.timedelta(lookback)


# In[12]:


data = pdr.get_data_yahoo(stocks,start)['Adj Close']
data.head()


# In[13]:


stocks


# In[14]:


len(data)-1


# In[15]:


annualized_returns = (data.iloc[len(data)-1]/data.iloc[0])**((365/lookback))-1
annualized_returns


# In[16]:


daily_returns = data.pct_change(1).dropna()
daily_returns


# In[17]:


std_dev = np.std(daily_returns)*math.sqrt(252)


# In[18]:


sharpe_ratio = (annualized_returns-0)/std_dev
sharpe_ratio


# In[19]:


sharpe_ratio = sharpe_ratio.sort_values().dropna()
sharpe_ratio


# In[20]:


plt.plot(sharpe_ratio)


# In[ ]:




