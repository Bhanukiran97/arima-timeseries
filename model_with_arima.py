# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:59:29 2022

@author: BHANUKIRAN
"""

# Importing Necessary Libraries

import pandas as pd
import nsepy
from datetime import date
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Extracting Data from NSE
# Infosys
infosys_data = nsepy.get_history(symbol = "INFY", start = date(2012,10,1), end = date.today())
infosys_daily_data = infosys_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                        'Deliverable Volume','%Deliverble'],  axis = 1)
infosys_daily_data.index = pd.DatetimeIndex(infosys_daily_data.index)
infosys_daily_data = infosys_daily_data.asfreq('b')
infosys_daily_data.interpolate(method = 'linear', inplace = True)
infosys_daily_data.to_csv('infosys_daily_data.csv')

# Reliance
reliance_data = nsepy.get_history(symbol = "RELIANCE", start = date(2012,10,1), end = date.today())
reliance_daily_data = reliance_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                          'Deliverable Volume','%Deliverble'],  axis = 1)
reliance_daily_data.index = pd.DatetimeIndex(reliance_daily_data.index)
reliance_daily_data = reliance_daily_data.asfreq('b')
reliance_daily_data.interpolate(method = 'linear', inplace = True)
reliance_daily_data.to_csv('reliance_daily_data.csv')

# Tata Motors
tatamotors_data = nsepy.get_history(symbol = "TATAMOTORS", start = date(2012,10,1), end = date.today())
tatamotors_daily_data = tatamotors_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                              'Deliverable Volume','%Deliverble'],  axis = 1)
tatamotors_daily_data.index = pd.DatetimeIndex(tatamotors_daily_data.index)
tatamotors_daily_data = tatamotors_daily_data.asfreq('b')
tatamotors_daily_data.interpolate(method = 'linear', inplace = True)
tatamotors_daily_data.to_csv('tatamotors_daily_data.csv')

# Wipro
wipro_data = nsepy.get_history(symbol = "WIPRO", start = date(2012,10,1), end = date.today())
wipro_daily_data = wipro_data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
                                    'Deliverable Volume','%Deliverble'],  axis = 1)
wipro_daily_data.index = pd.DatetimeIndex(wipro_daily_data.index)
wipro_daily_data = wipro_daily_data.asfreq('b')
wipro_daily_data.interpolate(method = 'linear', inplace = True)
wipro_daily_data.to_csv('wipro_daily_data.csv')


# Model Building and Saving

infosys_model_arima = ARIMA(infosys_daily_data['Close'], order = (1,2,1)).fit()
pickle.dump(infosys_model_arima, open('infosys_model_arima.sav', 'wb'))

reliance_model_arima = ARIMA(reliance_daily_data['Close'], order = (1,2,1)).fit()
pickle.dump(reliance_model_arima, open('reliance_model_arima.sav', 'wb'))
    
tatamotors_model_arima = ARIMA(tatamotors_daily_data['Close'], order = (1,1,1)).fit()
pickle.dump(tatamotors_model_arima, open('tatamotors_model_arima.sav', 'wb'))
    
wipro_model_arima = ARIMA(wipro_daily_data['Close'], order = (1,2,1)).fit()
pickle.dump(wipro_model_arima, open('wipro_model_arima.sav', 'wb'))