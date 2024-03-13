import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def stacked_lines_plots_w_anomalies(df, x_axis='datetime', metrics=None, anomaly_indices=[]):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(20, 10), sharex=True)
    
    for i , metric in enumerate(metrics):
        axes[i].plot(df.index, df[metric], label=metric)
        axes[i].set_title(f'{metric}')
        axes[i].set_ylabel(f'{metric}')
        
        for anomaly in anomaly_indices:
            axes[i].axvline(x=anomaly, color='red', linewidth=1, linestyle='--', label='Anomaly')
            
    axes[-1].set_xlabel(x_axis)
    ticks_to_use = df.index[::200]
    plt.xticks(ticks_to_use, rotation=45)
    
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.show()