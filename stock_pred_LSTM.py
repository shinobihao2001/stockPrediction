import logging
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Define the number of days for prediction
prediction_days = 60


def train_model(scaler_data, train_inputs):
    close_scaled = scaler_data["Close"]
    roc_scaled = scaler_data["ROC"]
    switch_train = {
        "Close": close_scaled,
        "ROC": roc_scaled
    }
    scaled_data = close_scaled
    if train_inputs["num_of_inputs"] == 1:
        scaled_data = switch_train[train_inputs["input_type"]]
    if train_inputs["num_of_inputs"] == 2:
        scaled_data = np.concatenate((close_scaled, roc_scaled), axis=1)
    # Prepare the training data
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    return lstm_model

def predict_candle_price(lstm_model, df, scaler, train_inputs):

    close_scaled = scaler["Close"].transform(df["Close"].values[-prediction_days:].reshape(-1, 1))
    roc_scaled = scaler["ROC"].transform(df["ROC"].values[-prediction_days:].reshape(-1, 1))
    switch_train = {
        "Close": close_scaled,
        "ROC": roc_scaled
    }
    scaled_data = close_scaled
    if train_inputs["num_of_inputs"] == 1:
        scaled_data = switch_train[train_inputs["input_type"]]
    if train_inputs["num_of_inputs"] == 2:
        scaled_data = np.concatenate((close_scaled, roc_scaled), axis=1)
    x_predict = []
    x_predict.append(scaled_data)
    x_predict = np.array(x_predict)
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 1))
    predicted_price = lstm_model.predict(x_predict)
    predicted_price = scaler["Close"].inverse_transform(predicted_price)

    return predicted_price


