import logging
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN

# Define the number of days for prediction
prediction_days = 60


def train_model(scaled_data):
    # Prepare the training data
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the RNN model
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    rnn_model.add(SimpleRNN(units=50))
    rnn_model.add(Dense(1))
    rnn_model.compile(loss="mean_squared_error", optimizer="adam")
    rnn_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    return rnn_model

def predict_candle_price(rnn_model, df, scaler):
    last_data = df["Close"].values[-prediction_days:].reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)
    x_predict = []
    x_predict.append(last_data_scaled)
    x_predict = np.array(x_predict)
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 1))
    predicted_price = rnn_model.predict(x_predict)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price
