import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def stacked_line_plots(df, x='time', metrics=None, title="Plot"):
    """
    A function to create stacked line plots for the given dataframe and metrics.
    
    Parameters:
    df: DataFrame
        The input dataframe containing the data to be plotted.
    x: str
        The column name to be used as the x-axis for the plots which should be present in the dataframe.
    metrics: list
        The list of column names to be plotted as stacked line plots.
    title: str
        The title for the plot.
    """

    if not metrics:
        raise ValueError('Please provide metrics to plot')
    # Create subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(20, 20), sharex=True)
    plt.title(title)
    # Loop through metrics and plot
    for i, metric in enumerate(metrics):
        axes[i].plot(df[x], df[metric], label=f'{metric} Data')
        #axes[i].scatter(anomaly_times, pm_data[metric].iloc[original_anomalous_indexes], color='red', label='Anomaly')
        axes[i].set_title(f'{metric}',fontsize=8)
        axes[i].set_ylabel(metric ,fontsize=8)


    # Label the shared x-axis
    axes[-1].set_xlabel(x.upper())
    ticks_to_use = df[x][::30]  # Choose every 10th time point
    plt.xticks(ticks_to_use)
    plt.xticks(rotation=45)

    # Format the dates on x-axis
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)

    fig.suptitle(title)
    # Show plot
    plt.show()
    
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