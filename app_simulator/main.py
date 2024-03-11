import streamlit as st
import numpy as np
from sample_generator import stock_data_generator
import requests
import json
import matplotlib.pyplot as plt
import time
import datetime

st.title('Stock Data Predictor')


col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date",  datetime.date(2023, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.date(2023, 1, 2))

st.warning('Please wait for predictions to complete...')

def create_windows(data, window_size, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def get_data():
    return stock_data_generator(start_date=start_date,
                    end_date=end_date,
                    freq='5min',
                    perc_anomalies=0.08,
                    regenerate=False,
                    write=False
                    )




df = get_data()
columns = df.columns.to_list()

all_metrics_data = df[columns].to_numpy()
overall_mean, overall_std = np.mean(all_metrics_data, axis=0), np.std(all_metrics_data, axis=0)

X, y = create_windows(df.to_numpy(), 24)

# Add prediction columns
for col in columns:
    df[col+'_pred'] = np.nan
    
fig, ax = plt.subplots()


ax.set_ylim(-50, 50)

actual_lines = []
prediction_lines =[]

for col in columns:
    actual_lines.append(ax.plot(df.index, df[col], label=col)[0])
    prediction_lines.append(ax.plot(df.index, df[col+'_pred'], label=col+' prediction')[0])






def init():  # give a clean slate to start
    for line in prediction_lines:
        line.set_ydata([np.nan] * len(df.index))

chunk_start = 0
chunk_target = 24

def animate(i):  # update the y values (every 1000ms)
    global chunk_start, chunk_target
    r = requests.post('http://torch_server:8000/predict', data=json.dumps({'input_data':X[chunk_start].tolist(), 'mean': overall_mean.tolist(), 'std': overall_std.tolist()}))
    predictions = json.loads(r.text)['prediction'][0]
    
    for col, pred, act_line, pred_line in zip(columns, predictions, actual_lines, prediction_lines):
    
        df.at[df.index[chunk_target], col+'_pred'] = pred
        act_line.set_ydata(df[col].mask(df.index > df.index[chunk_target]).tolist())
        pred_line.set_ydata(df[col+'_pred'].tolist())
        
    chunk_start+=1
    chunk_target+=1

    plt.legend(handles=actual_lines + prediction_lines, loc='upper left')
    plt.xticks(rotation=45)

    the_plot.pyplot(plt)

with st.spinner('Predicting...'):
    the_plot = st.pyplot(plt)
    init()
    for i in range(len(X)):
        animate(i)
        time.sleep(0.05)
        
    st.success('Done!')