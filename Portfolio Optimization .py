#!/usr/bin/env python
# coding: utf-8

# # Portfolio Optimization (Some Big Tech Stocks)

# In[1]:


# Importing libraries 
from pandas_datareader import data as web
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime 
plt.style.use('ggplot')


# In[2]:


# Ticker symbols in the portfolio 
assets = ['FB','AMZN',"AAPL","NFLX","GOOG"]


# In[3]:


# Weight assignments to stocks 
weights = np.array([0.2,0.2,0.2,0.2,0.2])


# In[4]:


# Portfolio Starting Date 
StartDate = '2015-06-06'
EndDate = '2022-11-08' 


# In[5]:


# Dataframe Creation 
df = pd.DataFrame()

for stock in assets: 
    df[stock]= web.DataReader(stock,data_source='yahoo',
                             start=StartDate,end=EndDate)['Adj Close']


# In[6]:


# Dataframe
df
df = df.dropna()
df


# In[7]:


title = 'Portfolio Adj, Close Price History'
my_stocks = df 

for c in my_stocks.columns.values: 
    plt.plot(my_stocks[c], label=c)
plt.title(title)
plt.xlabel("Date",fontsize=18)
plt.ylabel("Adj. Price USD",fontsize=18)
plt.legend(my_stocks.columns.values,loc='upper left')
plt.show()


# In[8]:


returns = df.pct_change()
returns 


# In[9]:


# Annualized Covariance Matrix 
cov_matrix_ann = returns.cov()*252
cov_matrix_ann


# In[10]:


# Calculation of the portfolio variance 
port_var = np.dot(weights.T, np.dot(cov_matrix_ann, weights))
port_var


# In[11]:


# Calculation of the portfolio volatility (Standard Deviation)
port_volatility = np.sqrt(port_var)
port_volatility


# In[22]:


# Calculation of the Annual Portfolio Return 
ann_port_return_simple = np.sum(returns.mean()*weights)*252
ann_port_return_simple


# In[23]:


# Expected Annual Return, Volatility (Risk) , & Variance 
percent_var = str(round(port_var,2)*100)+'%'
percent_vol = str(round(port_volatility,2)*100)+'%'
percent_return = str(round(ann_port_return_simple,2)*100)+'%'
print("Expected Annual Return: " + percent_var)
print("Annual Volatilitty / Risk :" +percent_vol)
print("Annual Variance: " + percent_return)


# In[24]:


from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models
from pypfopt import expected_returns 


# In[27]:


# Portfolio Optimization 
# Calculation of the Expected Returns 
# And Annualized Sample Covariance Matrix of the returns 

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize for max sharpe ratio 
ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[ ]:





# In[ ]:




