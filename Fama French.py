#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas_datareader.data as pdr
import pandas as pd
import datetime as dt
import statsmodels.api as sm


# In[107]:


end = dt.date(2022,10,30)
start = dt.date(end.year -10, end.month, end.day)
fund = ['FXAIX']


# In[108]:


fundsret = pdr.get_data_yahoo(fund,start,end)["Adj Close"].pct_change().dropna()


# In[109]:


fundsret_mtl = fundsret.resample('M').agg(lambda x: (x+1).prod() -1)
#cumulative returns on monthly basis


# In[110]:


fundsret_mtl


# In[111]:


factors = pdr.DataReader('F-F_Research_Data_Factors','famafrench',start,end)[0]


# In[112]:


factors


# In[113]:


#to check monthly asset returns to factors df, ensure both are the same
#fundsret_mtl.shape
factors.shape


# In[114]:


factors = factors[1:]


# In[115]:


factors.shape


# In[116]:


fundsret_mtl.shape


# In[117]:


fundsret_mtl.index = factors.index #match date formatting


# In[118]:


fundsret_mtl


# In[119]:


merge = pd.merge(fundsret_mtl,factors,on='Date')


# In[120]:


merge


# In[121]:


merge[['Mkt-RF','SMB','HML','RF']] = merge[['Mkt-RF','SMB','HML','RF']]/100


# In[122]:


merge


# In[123]:


merge['FXAIX-RF'] = merge.FXAIX - merge.RF


# In[124]:


merge


# In[125]:


y = merge['FXAIX-RF']
x = merge[['Mkt-RF','SMB','HML']]


X_stat = sm.add_constant(x)


# In[126]:


model = sm.OLS(y,X_stat)
results = model.fit()
results.summary()


# In[ ]:




