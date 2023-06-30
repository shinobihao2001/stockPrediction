import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb



def train_model(scaled_data):
    #prepare data to train
    x_train=scaled_data
    y_train=scaled_data[['Close']]
    #prepare model
    model= xgb.XGBRegressor();
    model.fit(x_train,y_train)
    return model

def predict_candle_price(model, df, scaler):
    last_data = df["Close"].values[-1:].reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)
    x_predict = []
    x_predict.append(last_data_scaled)
    x_predict = np.array(x_predict)
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 1))
    predicted_price = model.predict(x_predict)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price
