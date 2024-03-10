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
    end_date = st.date_input("Start Date", datetime.date(2023, 2, 1))



def create_windows(data, window_size, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
    
    
# @st.cache_data
def get_data():
    return stock_data_generator(start_date=start_date,
                    end_date=end_date,
                    perc_anomalies=0.08,
                    regenerate=False,
                    write=False
                    )

df = get_data()
columns = df.columns.to_list()
all_metrics_data = df[columns].to_numpy()

overall_mean, overall_std = np.mean(all_metrics_data, axis=0), np.std(all_metrics_data, axis=0)

X, y = create_windows(df.to_numpy(), 24)

for col in columns:
    df[col+'_pred'] = np.nan
    


fig, ax = plt.subplots()

x = df.index
ax.set_ylim(-50, 50)


line1, = ax.plot(x, df['stock_1'], label='stock 1')
line2, = ax.plot(x, df['stock_2'], label='stock 2')

line1_pred, = ax.plot(x, df['stock_1_pred'], label='stock 1 prediction')
line2_pred, = ax.plot(x, df['stock_2_pred'], label='stock 2 prediction')

the_plot = st.pyplot(plt)



def init():  # give a clean slate to start
    line1_pred.set_ydata([np.nan] * len(x))
    line2_pred.set_ydata([np.nan] * len(x))

chunk_start = 0
chunk_target = 24

def animate(i):  # update the y values (every 1000ms)
    global chunk_start, chunk_target
    r = requests.post('http://app:8000/predict', data=json.dumps({'input_data':X[chunk_start].tolist(), 'mean': overall_mean.tolist(), 'std': overall_std.tolist()}))
    predictions = json.loads(r.text)['prediction'][0]
    for col, pred in zip(columns, predictions):
        df.at[df.index[chunk_target], col+'_pred'] = pred
    
    line1.set_ydata(df['stock_1'].mask(df.index > df.index[chunk_target]).tolist())
    line2.set_ydata(df['stock_2'].mask(df.index > df.index[chunk_target]).tolist())
    # line2.set_ydata(df['stock_2'][0:chunk_target].tolist())
    
    line1_pred.set_ydata(df['stock_1_pred'].tolist())
    line2_pred.set_ydata(df['stock_2_pred'].tolist())
    chunk_start+=1
    chunk_target+=1

    plt.legend()
    plt.xticks(rotation=45)

    the_plot.pyplot(plt)

init()
for i in range(len(X)):
    animate(i)
    time.sleep(0.1)