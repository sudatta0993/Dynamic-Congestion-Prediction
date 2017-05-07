import numpy as np
import matplotlib.pyplot as plt

MINS_PER_DAY = 1440

def plot_io_curves(io_series, filepath, min_intervals):
    indices = np.arange(0, MINS_PER_DAY, min_intervals)
    for (input_series, output_series) in io_series:
        plt.plot(indices, input_series)
        plt.plot(indices, output_series)
    plt.ylabel('Cumulative number of vehicles')
    plt.xlabel('Time from midnight (mins)')
    plt.savefig(filepath)
    plt.close()

def plot_demand_congestion(demands, congestion, filepath, congestion_spillover = None, min_intervals = 5):
    indices = np.arange(0, MINS_PER_DAY - min_intervals, min_intervals)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if congestion_spillover is not None:
        ax2.plot(indices, congestion, 'b-',label='Zone c')
        ax2.plot(indices, congestion_spillover, 'r-', label='Zone a')
        ax2.legend()
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