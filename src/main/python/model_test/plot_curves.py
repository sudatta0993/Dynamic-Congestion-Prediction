import numpy as np
import matplotlib.pyplot as plt
import os

MINS_PER_DAY = 1440

def plot_fundamental_diagrams(parameters):
    for i in range(len(parameters.congestion_links_length)):
        ffs = parameters.congestion_links_length[i] / float(parameters.congestion_links_fftt[i])
        peak_flow_density = parameters.congestion_links_capacity[i] / ffs
        plt.plot([0,peak_flow_density],[0,parameters.congestion_links_capacity[i]],'b')
        plt.plot([peak_flow_density,parameters.congestion_links_jam_density[i]],[parameters.congestion_links_capacity[i],0],'b')
        plt.xlabel('Density (veh/km)')
        plt.ylabel('Flow (veh/min)')
        plt.savefig(parameters.file_directory + '/sample_plots/congestion_link_'+str(i)+'_FD.png')
        plt.close()
    for i in range(len(parameters.freeway_links_length)):
        ffs = parameters.freeway_links_length[i] / float(parameters.freeway_links_fftt[i])
        peak_flow_density = parameters.freeway_links_capacity[i] / ffs
        plt.plot([0, peak_flow_density], [0, parameters.freeway_links_capacity[i]], 'b')
        plt.plot([peak_flow_density, parameters.freeway_links_jam_density[i]],
                 [parameters.freeway_links_capacity[i], 0], 'b')
        plt.xlabel('Density (veh/km)')
        plt.ylabel('Flow (veh/min)')
        plt.savefig(parameters.file_directory + '/sample_plots/freeway_link_' + str(i) + '_FD.png')
        plt.close()

def plot_io_curves(io_series, filepath, min_intervals,toll_series = None,implement_tolls=False):
    indices = np.arange(0, MINS_PER_DAY, min_intervals)
    fig, ax1 = plt.subplots()
    for (input_series, output_series) in io_series:
        ax1.plot(indices, input_series,label='Before toll')
        ax1.plot(indices, output_series,label='After toll')
    if implement_tolls:
        ax2 = ax1.twinx()
        ax2.plot(indices, toll_series, '-r')
        ax2.set_ylabel('Toll values ($)', color='r')
        ax1.legend(loc=4)
    ax1.set_ylabel('Cumulative number of vehicles')
    ax1.set_xlabel('Time from midnight (mins)')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    plt.savefig(filepath)
    plt.close()

def plot_demand_congestion(demands, congestion, filepath, congestion_spillover = None, min_intervals = 5, num_bins = 288):
    indices = np.arange(0, num_bins*min_intervals - min_intervals, min_intervals)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if congestion_spillover is not None:
        ax2.plot(indices, congestion, 'b-',label='Zone c')
        ax2.plot(indices, congestion_spillover, 'r--', label='Zone a')
        ax2.legend()
        ax2.set_ylim([0,120])
    else:
        demand_colors = ['y:', 'g-.', 'r--']
        zone_dict = {0: 'a', 1: 'b', 2: 'd'}
        for i in range(len(demands)):
            demand_per_unit_time = [demands[i][j]/float(min_intervals) for j in range(len(demands[i]))]
            ax1.plot(indices, demand_per_unit_time, demand_colors[i],label='zone '+zone_dict.get(i)+' demand')
        ax2.plot(indices, congestion, '-')
        ax1.set_ylabel('Demand (num/min)')
    ax1.set_xlabel('Time from midnight (mins)')
    ax2.set_ylabel('MCL (min)', color='b')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    y_lim_max = [max(demands[i])/float(min_intervals) for i in range(len(demands))]
    ax1.set_ylim([0,max(y_lim_max) + 10])
    ax2.set_ylim([0,max(congestion)+30])
    ax1.legend(loc=1)
    plt.savefig(filepath)
    plt.close()