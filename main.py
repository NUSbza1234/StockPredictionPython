import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import datetime

# Set up the Streamlit app
st.title('Stock Price Prediction App')
st.write('Enter the stock ticker to get the stock price prediction.')

# Create a text input for the stock ticker
ticker_input = st.text_input('Stock Ticker', 'AAPL')  # Default value is AAPL

# Create a date input for the start and end dates
start_date = st.date_input('Start Date', datetime.date(2020, 1, 1))
end_date = st.date_input('End Date', datetime.date.today())

# Fetch and prepare the data if the button is pressed
if st.button('Predict'):
    # Fetch the data from Yahoo Finance
    data = yf.download(ticker_input, start=start_date, end=end_date)

    if not data.empty:
        # Use the 'Close' price for prediction
        data = data[['Close']]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Create training and test sets
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]

        # Create the dataset with timesteps
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        # Reshape into X=t,t+1,t+2,...t+n and Y=t+1
        time_step = 60
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape input to be [samples, time steps, features] for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))  # Output layer

        # Compile and fit the model with early stopping
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1, callbacks=[early_stopping])

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform to get actual prices
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot the results
        plt.figure(figsize=(14, 5))
        plt.plot(data.index, data, label='Original Data')
        plt.plot(data.index[time_step:time_step + len(train_predict)], train_predict, label='Train Prediction')
        plt.plot(data.index[time_step + len(train_predict):time_step + len(train_predict) + len(test_predict)], test_predict, label='Test Prediction')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Prediction for {ticker_input}')

        st.pyplot(plt)
    else:
        st.write('No data found for the ticker. Please check the ticker and try again.')

