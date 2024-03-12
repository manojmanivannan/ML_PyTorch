import streamlit as st
import numpy as np
from sample_generator import stock_data_generator
import requests
import json
import matplotlib.pyplot as plt
import time
import datetime
import imageio.v2 as imageio
import io

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


with open('model_data.json', 'r') as f:
    model_metadata = json.load(f)
window_size = model_metadata['window_size']


df = get_data()
columns = df.columns.to_list()

all_metrics_data = df[columns].to_numpy()
overall_mean, overall_std = np.mean(all_metrics_data, axis=0), np.std(all_metrics_data, axis=0)

X, y = create_windows(df.to_numpy(), window_size)

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




def animate(i):  # update the y values (every 1000ms)
    global window_size
    r = requests.post('http://torch_server:8080/predictions/onnx', json=json.dumps({"input": X[i].tolist(),  'mean': overall_mean.tolist(), 'std': overall_std.tolist()}), headers={'Content-Type': 'application/json'})
      
    print(r.text)
    predictions = json.loads(r.text)
    
    for col, pred, act_line, pred_line in zip(columns, predictions, actual_lines, prediction_lines):
    
        df.at[df.index[i+window_size], col+'_pred'] = pred
        act_line.set_ydata(df[col].mask(df.index > df.index[i+window_size]).tolist())
        pred_line.set_ydata(df[col+'_pred'].tolist())
        

    plt.legend(handles=actual_lines + prediction_lines, loc='upper left')
    plt.xticks(rotation=45)

    the_plot.pyplot(plt)

with st.spinner('Predicting...'):
    the_plot = st.pyplot(plt)
    init()
    images = []
    for i in range(len(X)):
        animate(i)
        # time.sleep(0.05)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        
    # Save images as GIF
    imageio.mimsave("plot.gif", images)
    
    st.success('Done!')

with open('plot.gif', 'rb') as f:
   if st.download_button('Download GIF', f, file_name='plot.gif'):
       st.stop()
