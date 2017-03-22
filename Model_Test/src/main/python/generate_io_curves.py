import numpy as np
import pandas as pd

MINS_PER_DAY = 1440

def get_congestion_links_input_curve_from_demand(num_zones, od_demand_funcs,min_intervals):
    congestion_links_input_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY,min_intervals))
    for i in range(num_zones):
        count = 0
        cum_arrivals = 0
        cum_arrival_curve = []
        for t in range(0, MINS_PER_DAY, min_intervals):
            cum_arrivals += sum([od_demand_funcs[i*num_zones+j](t) * min_intervals for j in range(num_zones)])
            cum_arrival_curve.append(cum_arrivals)
            count += 1
        congestion_links_input_curve[i] = pd.Series(cum_arrival_curve, index=np.arange(0, MINS_PER_DAY,min_intervals))
    return congestion_links_input_curve

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