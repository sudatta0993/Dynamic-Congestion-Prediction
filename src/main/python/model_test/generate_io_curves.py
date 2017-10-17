import numpy as np
import pandas as pd
from plot_curves import plot_io_curves

MINS_PER_DAY = 1440

def get_congestion_links_input_curve_from_demand(num_zones, od_demand_funcs,min_intervals):
    congestion_links_input_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY,min_intervals))
    for i in range(num_zones):
        cum_arrivals = 0
        cum_arrival_curve = []
        for t in range(0, MINS_PER_DAY, min_intervals):
            cum_arrivals += sum([od_demand_funcs[i*num_zones+j](t) * min_intervals for j in range(num_zones)])
            cum_arrival_curve.append(cum_arrivals)
        congestion_links_input_curve[i] = pd.Series(cum_arrival_curve, index=np.arange(0, MINS_PER_DAY,min_intervals))
    return congestion_links_input_curve

def get_congestion_links_input_curve_after_toll(congestion_links_input_curve_from_demand, toll_curves,
                                                value_of_time_early, value_of_time_late, num_zones, min_intervals,
                                                plot_cum_input_curves_toll,file_directory):
    congestion_links_input_curve_after_toll = pd.DataFrame(index=np.arange(0, MINS_PER_DAY, min_intervals))
    for i in range(num_zones - 1):
        cum_arrival_curve_from_toll = congestion_links_input_curve_from_demand[i].copy()
        cum_arrival_curve_from_demand = congestion_links_input_curve_from_demand[i].copy()
        toll_curve_values = [toll_curves[i](j) for j in np.arange(0, MINS_PER_DAY, min_intervals)]
        stationary_point_index = np.argmax(toll_curve_values)
        max_toll = np.max(toll_curve_values)
        toll_span = np.count_nonzero(np.diff(toll_curve_values))
        demand_span = np.count_nonzero(np.diff(cum_arrival_curve_from_demand))
        shift_ratio = (toll_span - demand_span)/float(demand_span) if (toll_span - demand_span)/float(demand_span) > 0 else 1
        past_index = 0
        for j in range(0,stationary_point_index):
            earlyness = shift_ratio*float(max_toll/value_of_time_late[i] - toll_curve_values[j]/value_of_time_early[i])
            new_index = max(j - int(earlyness / min_intervals), 0)
            cum_arrival_curve_from_toll.iloc[new_index] = cum_arrival_curve_from_demand.iloc[j]
            # Interpolate values in between
            for k in range(min(past_index + 1,new_index), new_index):
                slope = (cum_arrival_curve_from_toll.iloc[new_index] - cum_arrival_curve_from_toll.iloc[past_index]) / (
                new_index - past_index)
                cum_arrival_curve_from_toll.iloc[k] = cum_arrival_curve_from_toll.iloc[past_index] + slope * (
                k - past_index)
            past_index = new_index
        for j in range(stationary_point_index,MINS_PER_DAY/min_intervals):
            delay = shift_ratio*float(max_toll/value_of_time_late[i] - toll_curve_values[j]/value_of_time_late[i])
            new_index = min(j + int(delay/min_intervals),MINS_PER_DAY/min_intervals - 1)
            cum_arrival_curve_from_toll.iloc[new_index] = cum_arrival_curve_from_demand.iloc[j]
            # Interpolate values in between
            for k in range(past_index + 1,new_index):
                slope = (cum_arrival_curve_from_toll.iloc[new_index] - cum_arrival_curve_from_toll.iloc[past_index])/(new_index - past_index)
                cum_arrival_curve_from_toll.iloc[k] = cum_arrival_curve_from_toll.iloc[past_index] + slope * (k - past_index)
            past_index = new_index
        congestion_links_input_curve_after_toll[i] = pd.Series(cum_arrival_curve_from_toll, index=np.arange(0, MINS_PER_DAY,min_intervals))
        if plot_cum_input_curves_toll:
            plot_io_curves([(congestion_links_input_curve_from_demand[i], congestion_links_input_curve_after_toll[i])],
                           file_directory+'/sample_plots/input_curve_toll_zone_'+str(i)+'.png',min_intervals,toll_curve_values, True)
    return congestion_links_input_curve_after_toll


def get_freeway_links_input_curve_after_diverging(congestion_links_output_curve, num_zones,min_intervals, num_bins):
    freeway_links_input_curve = pd.DataFrame()
    for i in range(num_zones):
        for j in range(num_zones):
            if j == num_zones - 1 and j != i:
                freeway_links_input_curve[i * num_zones + j] = congestion_links_output_curve[i].copy()
            else:
                freeway_links_input_curve[i * num_zones + j] = pd.Series(np.zeros(num_bins),
                                                                index=np.arange(0, MINS_PER_DAY, min_intervals))
    return freeway_links_input_curve

def get_congestion_links_input_curve_after_merging(freeway_links_output_curve, num_zones,min_intervals):
    congestion_links_input_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY, min_intervals))
    for j in range(num_zones):
        for i in range(num_zones):
            if j in congestion_links_input_curve:
                congestion_links_input_curve[j] = congestion_links_input_curve[j].values + \
                                                freeway_links_output_curve[i * num_zones + j].copy().values
            else:
                congestion_links_input_curve[j] = freeway_links_output_curve[i * num_zones + j].values
    return congestion_links_input_curve

def get_links_output_curve(links_input_curve, links_capacity, links_fftt,min_intervals, num_bins):
    links_output_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY, min_intervals))
    i = 0
    for column in links_input_curve:
        cum_arrival_curve = links_input_curve[column]
        capacity = links_capacity[i]
        fftt = links_fftt[i]
        cum_departure_curve = cum_arrival_curve.copy()
        for j in range(num_bins):
            if cum_arrival_curve.iloc[j] > cum_departure_curve.iloc[j-1] + capacity * min_intervals:
                cum_departure_curve.iloc[j] = cum_departure_curve.iloc[j - 1] + capacity * min_intervals
        cum_departure_curve = cum_departure_curve.shift(int(fftt / min_intervals)).fillna(0)
        links_output_curve[column] = cum_departure_curve.values
        i += 1
    return links_output_curve