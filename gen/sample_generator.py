import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy.signal import savgol_filter



def data_generator(start_date: tuple, end_date: tuple, freq: str='5min', perc_anomalies: float = 5, regenerate: bool = False, write: bool = True):
    """
    Returns a pandas dataframe of a timeseries data 2 metrics and 1 dimensions
    and writes as csv. If a csv already exists with the same specification, reads
    the csv and returns it.
    
    Parameters:
        start_date:     tuple (year, month, date)
        end_date:       tuple (year, month, date)
        freq:           frequency of timeseries. default=5T
        perc_anomalies: Percentage of data points that are anomalous. default=5
    
    Return:
        pandas dataframe
    """
    def create_datetime(date: tuple):
        return datetime(date[0],date[1],date[2])
    
    start = create_datetime(start_date)
    end   = create_datetime(end_date)
    
    csv_filename = "_".join([
                    'trig',
                    'start_' + start.strftime('%y_%m_%d'),
                    'end_'   + end.strftime('%y_%m_%d'),
                    'freq_'  + freq,
                    'perc_'  + str(perc_anomalies)
    ]) + '.csv'
    file_path = os.path.abspath(os.path.join(os.getcwd(), 'data', csv_filename))

    if os.path.isfile(file_path) and not regenerate:
        print(f'Loading from filesystem {file_path}')
        return pd.read_csv(file_path,parse_dates=['datetime'],index_col='datetime')
    

    datetime_range = pd.date_range(start,end,freq=freq)

    # Generate sine and cosine values with noise
    num_samples = len(datetime_range)
    
    # Define frequencies for the sine waves
    sin_freqs = [1, 4, 0.5/3, 16/3, 2, 8/5]
    cos_freqs = [2, 3, 0.5/4, 17/4, 5, 9/7]

    # Initialize the complex sine wave
    sine_wave = np.zeros(num_samples)
    cosine_wave = np.zeros(num_samples)

    # Generate individual sine waves and combine them
    for freq in sin_freqs:
        wave = np.sin(np.linspace(0, 2 * np.pi * freq, num_samples))
        sine_wave += wave
    
    for freq in cos_freqs:
        wave = np.sin(np.linspace(0, 2 * np.pi * freq, num_samples))
        cosine_wave += wave
    
    

    # generating noise 
    
    sine_noise = np.random.normal(0, 0.1, num_samples)
    cosine_noise = np.random.normal(0, 0.1, num_samples)
    
    sine_values = sine_wave + sine_noise
    cosine_values = cosine_wave + cosine_noise
    
    
    # smoothen the noise
    
    sine_values = savgol_filter(sine_values, 11, 2)
    cosine_values = savgol_filter(cosine_values,11, 2)
    
    # add anomalies
    
    num_anomalies = int((perc_anomalies / 100) * num_samples)
    sin_anomaly_indices = np.random.choice(np.arange(num_samples), num_anomalies, replace=False)
    cos_anomaly_indices = np.random.choice(np.arange(num_samples), num_anomalies, replace=False)
   
    # Modify anomalies based on the mean or standard deviation
    sine_mean = np.mean(sine_values)
    sine_std = np.std(sine_values)
    sine_values[sin_anomaly_indices] += 1 * sine_std
    
    cosine_mean = np.mean(cosine_values)
    cosine_std = np.std(cosine_values)
    cosine_values[cos_anomaly_indices] += 1 * cosine_std
   
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': datetime_range,
        'sine': sine_values,
        'cosine': cosine_values
    })
    print('Generating data: '+file_path)
    df.set_index('datetime',drop=True, inplace=True)
    if write:
        df.to_csv(file_path)
    return df

def stacked_data_generator(start_date: tuple, end_date: tuple, freq: str='5T', perc_anomalies: float = 5, regenerate: bool = False):
    
    dimensions = ['A','B','C']
    perc_anomalies_list = [0.02,0.05,0.03]
    stack = []
    for dim,perc in zip(dimensions, perc_anomalies_list):
        df = data_generator(start_date, end_date, freq, perc, regenerate)
        df['dimension'] = dim # np.random.choice(['a', 'b', 'c'], len(df))
        stack.append(df)
        
    df =pd.concat(stack)
    
    return df.sample(frac=1)


def stock_data_generator(start_date, end_date, freq: str='5T', perc_anomalies: float=5, regenerate: bool = False, write: bool = True):
    """
    Returns a pandas dataframe of a timeseries data with stock price-like columns
    and writes as csv. If a csv already exists with the same specification, reads
    the csv and returns it.
    
    Parameters:
        start_date:     tuple (year, month, date)
        end_date:       tuple (year, month, date)
        freq:           frequency of timeseries. default=5T
        perc_anomalies: Percentage of data points that are anomalous. default=5
    
    Return:
        pandas dataframe
    """
    def create_datetime(date):
        if isinstance(date, tuple):
            return datetime(date[0], date[1], date[2])
        return date
    
    start = create_datetime(start_date)
    end = create_datetime(end_date)
    
    csv_filename = "_".join([
                    'stock',
                    'start_' + start.strftime('%y_%m_%d'),
                    'end_' + end.strftime('%y_%m_%d'),
                    'freq_' + freq,
                    'perc_' + str(perc_anomalies)
    ]) + '.csv'
    file_path = os.path.abspath(os.path.join(os.getcwd(), 'data', csv_filename))

    if os.path.isfile(file_path) and not regenerate:
        print(f'Loading from filesystem {file_path}')
        return pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
    
    
    datetime_range = pd.date_range(start, end, freq=freq)

    # Simulate stock price-like columns
    num_samples = len(datetime_range)
    num_stocks = 2  # Number of stock-like columns
    
    stock_prices = np.zeros((num_samples, num_stocks))
    for i in range(num_stocks):
        stock_prices[:, i] = np.cumsum(np.random.randn(num_samples)) + np.sin(np.linspace(0, 2 * np.pi * 17/3, num_samples))  # Random walk process
    
    # Add noise
    stock_prices += np.random.normal(0, 0.1, size=(num_samples, num_stocks))
    
    # Apply smoothing
    for i in range(num_stocks):
        stock_prices[:, i] = savgol_filter(stock_prices[:, i], 11, 3)
    
    # Add anomalies
    num_anomalies = int((perc_anomalies / 100) * num_samples)
    
    anomaly_indices = np.zeros((num_anomalies, num_stocks),dtype=int)
    
    for i in range(num_stocks):
        anomaly_indices[:, i] = np.random.choice(np.arange(num_samples), num_anomalies, replace=False)
    

    
    for i in range(num_stocks):
        stock_mean = np.mean(stock_prices[:, i])
        stock_std = np.std(stock_prices[:, i])
        stock_prices[anomaly_indices[:,i], i] += 2 * stock_std  # Spike anomalies
    
    # Create DataFrame
    df = pd.DataFrame(stock_prices, columns=[f'stock_{i+1}' for i in range(num_stocks)], index=datetime_range)
    df.rename_axis('datetime',inplace=True)

    print('Generating data: '+file_path)
    if write:
        df.to_csv(file_path)
    return df

