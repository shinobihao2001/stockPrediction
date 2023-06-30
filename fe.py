import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import websocket
import json
from binance.spot import Spot as Client
import threading
import stock_pred_LSTM as lstm
import stock_pred_RNN as rnn

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
roc = df["Close"].pct_change() * 100
roc = roc.rename("ROC")
df["ROC"] = roc

# Create a scaler for normalization
scaler ={
    "Close": MinMaxScaler(feature_range=(0, 1)),
    "ROC": MinMaxScaler(feature_range=(0, 1))
}
scaled_data = {
    "Close": scaler["Close"].fit_transform(df["Close"].values.reshape(-1, 1)),
    "ROC": scaler["ROC"].fit_transform(df["ROC"].values.reshape(-1, 1))
}

lstm_model = None
rnn_model = None
selected_model = "LSTM"
predicted_candle = None
train_inputs = {
    "num_of_inputs": 1,
    "input_type": "Close",
}

# default train
lstm_model = lstm.train_model(scaled_data, train_inputs)
rnn_model = rnn.train_model(scaled_data, train_inputs)


def predicted_candlestick(modeltype, df, scaler, next_open_time, train_inputs):
    swithc_model = {
        "LSTM": {
            "model": lstm,
            "train_model": lstm_model
        },
        "RNN": {
            "model": rnn,
            "train_model": rnn_model
        }

    }
    model = swithc_model[modeltype]["model"]
    model_train = swithc_model[modeltype]["train_model"]
    predicted_price = model.predict_candle_price(model_train, df, scaler, train_inputs)
    predicted_price = predicted_price[0][0]
    current_price = df["Close"].values[-1]
    prredicted_candle = {
        "Open time": None,
        "Open": None,
        "High": None,
        "Low": None,
        "Close": None
    }
    prredicted_candle["Open time"] = next_open_time
    prredicted_candle["Open"] = current_price
    prredicted_candle["High"] = max(current_price, predicted_price)
    prredicted_candle["Low"] = min(current_price, predicted_price)
    prredicted_candle["Close"] = predicted_price
    return prredicted_candle

predicted_candle = predicted_candlestick(selected_model, df, scaler, 
                                            pd.to_datetime(df.index[-1], unit="ms") + pd.Timedelta(minutes=int(interval[:-1])), train_inputs)
# Set up the WebSocket connection
SOCKET = f'wss://stream.binance.com:9443/ws/{crypto_name.lower()}@kline_{interval}'
ws = websocket.WebSocketApp(SOCKET)

def on_message(ws, message):
    global df, predicted_candle
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
    roc = (float(close) - float(open)) / float(open) * 100
    new_data = pd.DataFrame(
        [[open, high, low, close, volume, roc]],
        columns=["Open", "High", "Low", "Close", "Volume", "ROC"],
        index=[opentime]
    )

    df = pd.concat([df, new_data])
    df = df.astype(float)
    df = df[~df.index.duplicated(keep='last')]
    if is_candle_closed:
        if (train_inputs != None):
            # Predict next crypto price
            predicted_candle = predicted_candlestick(algorithm, df, scaler,
                                            pd.to_datetime(df.index[-1], unit="ms") + pd.Timedelta(minutes=int(interval[:-1])), train_inputs)
            #print(prredicted_candle)


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
@app.callback(Output("crypto_chart", "figure"), [Input("update_interval", "n_intervals"), Input("algorithm", "value")])
def update_chart(n, algorithm):
    global predicted_candle, selected_model
    if(predicted_candle is None):
        return {}
    
    recent_data = df.tail(20)
    if algorithm != selected_model:
        df_no_tail = df[:-1]
        predicted_candle = predicted_candlestick(selected_model, df_no_tail, scaler, 
                                            pd.to_datetime(df.index[-1], unit="ms"),
                                            train_inputs)
        
        selected_model = algorithm

    predicted_trace = go.Candlestick(
            x=[predicted_candle["Open time"]],
            open=[predicted_candle["Open"]],
            high=[predicted_candle["High"]],
            low=[predicted_candle["Low"]],
            close=[predicted_candle["Close"]],
            name= "{} Predicted Candle".format(algorithm),
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
app.layout = html.Div(
    children=[
        html.H1(children="GIA HAO, XUAN HANH, LE DAT"),
        html.Div(children="'''\nDự đoán giá cổ phiếu\n'''"),
        html.Div(
            id="input-container",
            children=[
                html.Label("Đặc trưng dự đoán: "),
                dcc.Dropdown(
                    id="prediction",
                    options=[
                        {"label": "Close", "value": "Close"},
                        {"label": "ROC", "value": "ROC"},
                    ],
                    multi=True,
                    value=["Close"],
                    style={"width": 150},
                    clearable=False,
                ),
                html.Button("Huấn luyện", id="start", n_clicks=0),
                html.Div(id="output-container-button"),
            ],
        ),
        html.Div(
            id="chart-container",
            style={"display": "block"}, 
            children=[
                html.Div(
                    children=[
                        html.Label("Thuật toán: "),
                        dcc.Dropdown(
                            id="algorithm",
                            options=[
                                {"label": "XGBoost", "value": "XGBoost"},
                                {"label": "RNN", "value": "RNN"},
                                {"label": "LSTM", "value": "LSTM"},
                            ],
                            value="LSTM",
                            clearable=False,
                            searchable=False,
                            style={"width": 150},
                        ),
                    ]
                ),
                dcc.Graph(id="crypto_chart"),
                dcc.Interval(
                    id="update_interval",
                    interval=1 * 1000,  # Cập nhật mỗi 1 giây
                    n_intervals=0,
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("output-container-button", "children"),
    [Input("start", "n_clicks")],
    [State("prediction", "value")],
)
def update_output(n_clicks, value):
        global train_inputs
        if len(value) == 1:
            train_inputs = {"num_of_inputs": 1, "input_type": value[0]}
        else:
            train_inputs = {"num_of_inputs": len(value), "input_type": value}
        global lstm_model, rnn_model
        lstm_model = lstm.train_model(scaled_data, train_inputs)
        print("LSTM model trained")
        rnn_model = rnn.train_model(scaled_data, train_inputs)
        print("RNN model trained")
        global selected_model, predicted_candle
        predicted_candle = predicted_candlestick(selected_model, df, scaler,
                                                pd.to_datetime(df.index[-1], unit="ms") + pd.Timedelta(minutes=int(interval[:-1])),
                                                train_inputs)
        # Hiển thị huấn luyện đã hoàn thành
        return  "Huấn luyện hoàn tất"

if __name__ == "__main__":
    app.run_server(debug=True)