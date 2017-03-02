import numpy as np
import matplotlib.pyplot as plt

MINS_PER_DAY = 1440
MIN_INTERVALS = 5

def plot_io_curves(io_series, filepath):
    indices = np.arange(0, MINS_PER_DAY, MIN_INTERVALS)
    for (input_series, output_series) in io_series:
        plt.plot(indices, input_series)
        plt.plot(indices, output_series)
    plt.savefig(filepath)
    plt.close()

def plot_demand_congestion(demands, congestion, filepath, congestion_spillover = None):
    indices = np.arange(0, MINS_PER_DAY - MIN_INTERVALS, MIN_INTERVALS)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if congestion_spillover is not None:
        ax2.plot(indices, congestion, 'b-')
        ax2.plot(indices, congestion_spillover, 'r-')
        ax2.set_ylim([0,120])
    else:
        demand_colors = ['y-', 'g-', 'r-']
        for i in range(len(demands)):
            ax1.plot(indices, demands[i], demand_colors[i])
        ax2.plot(indices, congestion, 'b-')
        ax1.set_ylabel('Demand (num/min)')
    ax1.set_xlabel('Time from midnight (mins)')
    ax2.set_ylabel('Congestion (accumulation/output) (min)', color='b')
    plt.savefig(filepath)
    plt.close()