# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)


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
])

if __name__ == '__main__':
    app.run(debug=True)
