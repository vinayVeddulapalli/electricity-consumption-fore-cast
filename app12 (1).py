# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:24:01 2023

@author: govin
"""


import streamlit as st
import pandas as pd
import xgboost as xgb
import datetime
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("C:\\Users\\govin\\Our_Trained_model.sav", 'rb'))

# Create a Streamlit app
st.title('PJMW Forecasting')

# Add a date picker to select the start date for the forecast
start_date = st.date_input('Select a start date')

# Add a slider to select the number of days to forecast
num_days = st.slider('Number of days to forecast', 1, 30, 7)

# Load the data
data = pd.read_csv("C:\\Users\\govin\\columndata.csv", parse_dates=[0])

# Preprocess the data
#data = preprocess_data(data)

# Create a copy of the data with only the features needed for prediction
features = ['year', 'month', 'day', 'hour']
data_pred = data[features].copy()

# Create a list of dates to forecast
forecast_dates = pd.date_range(start=start_date, periods=num_days, freq='D')

# Add the forecast dates as new rows to the data
new_rows = [{'year': date.year, 'month': date.month, 'day': date.day, 'hour': date.hour} for date in forecast_dates]
data_pred = data_pred.append(new_rows, ignore_index=True)

# Add day of week, holiday, and peak hour features to the data
data_pred['day_of_week'] = pd.to_datetime(data_pred[['year', 'month', 'day']]).dt.dayofweek
data_pred['holiday'] = data_pred['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
data_pred['peak_hour'] = data_pred['hour'].apply(lambda x: 1 if x in [7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21] else 0)

# Apply one-hot encoding to the data
data_pred = pd.get_dummies(data_pred, columns=['year', 'month', 'day', 'hour', 'day_of_week','holiday','peak_hour'])

# Reorder columns to match the model
X_pred = data_pred.reindex(columns=model.get_booster().feature_names)

# Generate the forecast
y_pred = model.predict(X_pred)

# Display the result
if st.button('Generate Forecast'):
    st.subheader('Forecast Results')
    for i in range(num_days):
        date = forecast_dates[i].strftime("%Y-%m-%d")
        demand = round(y_pred[i], 2)
        st.write(f'The forecasted demand for PJMW on {date} is: {demand} MW')
print(data.head())
print(model)