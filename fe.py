import logging
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import websocket
import json
from binance.spot import Spot as Client
import threading

# Set up the logging configuration
# logging.basicConfig(level=logging.DEBUG)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Configure the Binance API client
base_url = "https://api.binance.com"
spot_client = Client(base_url=base_url)

# Define the cryptocurrency details
crypto_name = "BTCUSDT"
interval = "1m"
limit = 1000

# Retrieve cryptocurrency data from Binance
crypto_data = spot_client.klines(symbol=crypto_name, interval=interval, limit=limit)

# Convert data into a DataFrame
columns = [
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
]
df = pd.DataFrame(crypto_data, columns=columns)
df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
df.set_index("Open time", inplace=True)
df.drop(columns=["Close time", "Ignore", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume"], inplace=True)
df = df.astype(float)

# Create a scaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Define the number of days for prediction
prediction_days = 60

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

# Set up the WebSocket connection
SOCKET = f'wss://stream.binance.com:9443/ws/{crypto_name.lower()}@kline_{interval}'
ws = websocket.WebSocketApp(SOCKET)

prredicted_candle = {
    "Open time": 0,
    "Open": 0,
    "High": 0,
    "Low": 0,
    "Close": 0,
}

def on_message(ws, message):
    global df
    json_message = json.loads(message)
    candle = json_message['k']
    is_candle_closed = candle['x']

    opentime = candle['t']
    opentime = pd.to_datetime(opentime, unit="ms")
    open = candle['o']
    high = candle['h']
    low = candle['l']
    close = candle['c']
    volume = candle['v']

    new_data = pd.DataFrame(
        [[open, high, low, close, volume]],
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=[opentime]
    )

    df = pd.concat([df, new_data])
    df = df.astype(float)
    df = df[~df.index.duplicated(keep='last')]

    if is_candle_closed:
        # Predict next crypto price
        last_data = df["Close"].values[-prediction_days:].reshape(-1, 1)
        scaled_last_data = scaler.transform(last_data)
        x_predict = np.array([scaled_last_data])
        x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 1))
        predicted_data = lstm_model.predict(x_predict)
        predicted_price = scaler.inverse_transform(predicted_data)
        predicted_price = predicted_price[0][0]
        current_price = df["Close"].values[-1]
        prredicted_candle["Open time"] = pd.to_datetime(candle['T'], unit="ms")
        prredicted_candle["Open"] = current_price
        prredicted_candle["High"] = max(current_price, predicted_price)
        prredicted_candle["Low"] = min(current_price, predicted_price)
        prredicted_candle["Close"] = predicted_price
        print(prredicted_candle)


def on_close(ws):
    print("WebSocket connection closed")


ws.on_message = on_message
ws.on_close = on_close


def run_websocket():
    ws.run_forever()


# Create a new thread for WebSocket connection
ws_thread = threading.Thread(target=run_websocket)
ws_thread.start()

# # Update the candlestick chart in the Dash app
@app.callback(Output("crypto_chart", "figure"), [Input("update_interval", "n_intervals")])
def update_chart(n):
    recent_data = df.tail(10)
    predicted_trace = go.Candlestick(
        x=[prredicted_candle["Open time"]],
        open=[prredicted_candle["Open"]],
        high=[prredicted_candle["High"]],
        low=[prredicted_candle["Low"]],
        close=[prredicted_candle["Close"]],
        name="Predicted Candlestick",
        increasing_line_color="yellow",
        decreasing_line_color="orange"
    )
    fig = go.Figure(data=[
        go.Candlestick(
            x=recent_data.index,
            open=recent_data['Open'],
            high=recent_data['High'],
            low=recent_data['Low'],
            close=recent_data['Close'],
            name="Candlesticks"
        ),
    ])
    fig.add_trace(predicted_trace)
    return fig


# Set up the layout of the Dash app
app.layout = html.Div(children=[
    html.H1(children='GIA HAO, XUAN HANH, LE DAT'),

    html.Div(children='''
        Dự đoán giá cổ phiếu
    '''),

    html.Div(children=[html.Label("Thuật toán: "),
        dcc.Dropdown(['XGBoost','RNN','LSTM'],'LSTM',style={'width':150}),]
    ),

    html.Div(children=[html.Label("Giá trị dự đoán: "),
    dcc.RadioItems(['Close','ROC'],"Close")],
    ),

    dcc.Graph(id="crypto_chart"),
    dcc.Interval(
        id="update_interval",
        interval=1 * 1000,  # Update every 1 second
        n_intervals=0
    ),
])


if __name__ == "__main__":
    app.run_server(debug=True)
