import streamlit as st
import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

plt.style.use('fivethirtyeight')

# Download Stock Data
def get_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data[['Close']]

# Preprocess Data
def preprocess_data(data, train_ratio=0.8):
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * train_ratio)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler, dataset, training_data_len

# Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and Predict
def train_and_predict(stock_symbol, start_date, end_date):
    data = get_stock_data(stock_symbol, start_date, end_date)
    x_train, y_train, scaler, dataset, training_data_len = preprocess_data(data)

    model = build_lstm_model()
    model.fit(x_train, y_train, epochs=1, batch_size=1)

    # Create Testing Data
    test_data = scaler.transform(dataset)[training_data_len-60:, :]
    x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
    x_test = np.array(x_test).reshape(len(x_test), 60, 1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(((predictions - dataset[training_data_len:]) ** 2)))

    # Plot Results
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    return model, scaler

# Run the script
if __name__ == "__main__":
    model, scaler = train_and_predict('AAPL', '2012-01-01', '2019-12-17')
