# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:02:09 2022

@author: BHANUKIRAN
"""

import uvicorn
import pandas as pd
import streamlit as st
from datetime import date
from statsmodels.tsa.arima.model import ARIMA
from plotly import graph_objs as go
import pickle



infosys_model = pickle.load(open('infosys_model_arima.sav', 'rb'))
reliane_model = pickle.load(open('reliance_model_arima.sav', 'rb'))
tatamotors_model = pickle.load(open('tatamotors_model_arima.sav', 'rb'))
wipro_model = pickle.load(open('wipro_model_arima.sav', 'rb'))

model_dict = {'Infosys': infosys_model,
              'Reliance': reliane_model,
              'Tatamotors': tatamotors_model,
              'Wipro': wipro_model}


infosys_data = pd.read_csv('infosys_daily_data.csv')
reliane_data = pd.read_csv('reliance_daily_data.csv')
tatamotors_data = pd.read_csv('tatamotors_daily_data.csv')
wipro_data = pd.read_csv('wipro_daily_data.csv')



data_dict = {'Infosys': infosys_data,
              'Reliance': reliane_data,
              'Tatamotors': tatamotors_data,
              'Wipro': wipro_data
              }




st.title('Stock Forecast App')

st.subheader('Select Stock for Forecasting')
selected_stock = st.selectbox('Stock', model_dict.keys())


@st.cache
def load_data(x):
    return x

data = load_data(data_dict[selected_stock])


st.subheader(f'Past 10 Years Data for {selected_stock} Stock Price')
st.write(data)


# Plot raw data
st.subheader(f'Time Series Plot for {selected_stock} Stock Closing Price')

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()



st.subheader('Select No. of months for Forecasting')

n_months = st.slider('Slide to select', 1, 12)
period = n_months * 30

    
def arima(ticker):
    forecast_df = pd.DataFrame(ticker.forecast(steps = 261))
    forecast_df.reset_index(inplace =True)
    forecast_df.rename({'index':'Date','predicted_mean':'Close_forecast'}, axis = 1, inplace = True)
    forecast_df['Date'] = forecast_df['Date'].apply(lambda x: x.date())
        
    # plot forecast
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = "stock_close"))
    forecast_fig.add_trace(go.Scatter(x = forecast_df['Date'], y = forecast_df['Close_forecast'], name = "stock_close_forecast"))
    forecast_fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(forecast_fig)
    
    # Show forecast
    st.subheader(f'Forecast data for {n_months} months')
    st.write(forecast_df.head(n_months*30))
    
    
arima(model_dict[selected_stock])

