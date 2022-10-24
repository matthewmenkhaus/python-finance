#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
import datetime as dt


# In[96]:


stocks = "MSFT GOOG AMZN DPZ".split()
start = dt.date.today() - dt.timedelta(2000)


# In[97]:


data = pdr.get_data_yahoo(stocks,start)['Adj Close']
data.head()


# In[98]:


port_returns = (np.log(data).diff()).dropna()
port_returns.head()


# In[99]:


port_returns.plot(figsize=(12,3))


# In[100]:


plt.figure(dpi=150)
port_returns['MSFT'].plot(kind='hist',bins=100,figsize=(12,3))


# In[101]:


cumul_return = (1 + port_returns['MSFT']).cumprod() - 1
cumul_return_perc = cumul_return*100

cumul_return.plot()


# In[102]:


np.dot([2,3],[10,20])
example_returns = np.array([1,0.5]) # 100% and 50% returns
weights = [1,0]
np.dot(weights,example_returns)


# In[103]:


weights = [0.5,0.5]
np.dot(weights,example_returns)


# In[104]:


N = len(port_returns.columns)
equal_weights = N * [1/N]


# In[105]:


equal_weights


# In[106]:


equal_returns = np.dot(equal_weights,port_returns.transpose())


# In[107]:


equal_returns


# In[108]:


cum_equal_returns =   (1 + equal_returns).cumprod() - 1


# In[109]:


cum_equal_returns


# In[110]:


cum_equal_returns_perc = pd.Series(100 * cum_equal_returns)
cum_equal_returns_perc.index = port_returns.index


# In[111]:


plt.figure(dpi=150)
cum_equal_returns_perc.plot(figsize=(10,2))


# In[112]:


# What is better? All in MSFT or equal weights?

plt.figure(dpi=150)
cumul_return_perc.plot(figsize=(10,2),label='MSFT')
cum_equal_returns_perc.plot(figsize=(10,2),label='EQUAL WEIGHTS')
plt.legend()


# In[113]:


data/data.shift(1)


# In[114]:


log_rets = np.log(data/data.shift(1)).dropna()
log_rets


# In[115]:


data.pct_change(1).dropna()


# In[116]:


N = len(data.columns)

weights = np.random.random(N)


# In[117]:


weights


# In[118]:


np.sum(weights)


# In[119]:


weights = weights/ np.sum(weights)


# In[120]:


weights


# In[121]:


np.sum(weights)


# In[122]:


def gen_weights(N):
    weights = np.random.random(N)
    return weights/ np.sum(weights)


# In[123]:


def calculate_returns(weights,log_rets):
    return np.sum(log_rets.mean()*weights) * 252 #Annualized Returns


# In[124]:


log_rets_cov = log_rets.cov()


# In[125]:


def calculate_volatility(weights,log_rets_cov):
    annualized_cov = np.dot(log_rets_cov*252,weights)
    vol = np.dot(weights.transpose(),annualized_cov)
    return np.sqrt(vol)


# In[126]:


calculate_volatility(weights,log_rets_cov)


# In[127]:


log_rets = np.log(data/data.shift(1))
log_rets_cov = log_rets.cov()


# In[128]:


# This may take a while 

mc_portfolio_returns = []
mc_portfolio_vol = []
mc_weights = []
for sim in range(20000):
    weights = gen_weights(N)
    mc_weights.append(weights)
    mc_portfolio_returns.append(calculate_returns(weights,log_rets))
    mc_portfolio_vol.append(calculate_volatility(weights,log_rets_cov))


# In[129]:


mc_sharpe_ratios = np.array(mc_portfolio_returns)/np.array(mc_portfolio_vol)


# In[130]:


plt.figure(dpi=200,figsize=(10,5))
plt.scatter(mc_portfolio_vol,mc_portfolio_returns,c=mc_sharpe_ratios)
plt.ylabel('EXPECTDE RETS')
plt.xlabel('EXPECTED VOL')
plt.colorbar(label="SHARPE RATIO");


# In[131]:


mc_weights[np.argmax(mc_sharpe_ratios)]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




