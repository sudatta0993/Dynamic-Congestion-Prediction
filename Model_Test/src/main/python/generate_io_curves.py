import numpy as np
import pandas as pd

MINS_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MINS_PER_DAY / MIN_INTERVALS

def get_congestion_links_input_curve_from_demand(num_zones, od_demand_funcs):
    congestion_links_input_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY,MIN_INTERVALS))
    for i in range(num_zones):
        count = 0
        cum_arrivals = 0
        cum_arrival_curve = []
        for t in range(0, MINS_PER_DAY, MIN_INTERVALS):
            cum_arrivals += sum([od_demand_funcs[i*num_zones+j](t) * MIN_INTERVALS for j in range(num_zones)])
            cum_arrival_curve.append(cum_arrivals)
            count += 1
        congestion_links_input_curve[i] = pd.Series(cum_arrival_curve, index=np.arange(0, MINS_PER_DAY,MIN_INTERVALS))
    return congestion_links_input_curve

def get_freeway_links_input_curve_after_diverging(congestion_links_output_curve, num_zones):
    freeway_links_input_curve = pd.DataFrame()
    for i in range(num_zones):
        for j in range(num_zones):
            if j == num_zones - 1 and j != i:
                freeway_links_input_curve[i * num_zones + j] = congestion_links_output_curve[i].copy()
            else:
                freeway_links_input_curve[i * num_zones + j] = pd.Series(np.zeros(NUM_BINS),
                                                                index=np.arange(0, MINS_PER_DAY, MIN_INTERVALS))
    return freeway_links_input_curve

def get_congestion_links_input_curve_after_merging(freeway_links_output_curve, num_zones):
    congestion_links_input_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY, MIN_INTERVALS))
    for j in range(num_zones):
        for i in range(num_zones):
            if j in congestion_links_input_curve:
                congestion_links_input_curve[j] = congestion_links_input_curve[j].values + \
                                                freeway_links_output_curve[i * num_zones + j].copy().values
            else:
                congestion_links_input_curve[j] = freeway_links_output_curve[i * num_zones + j].values
    return congestion_links_input_curve

def get_links_output_curve(links_input_curve, links_capacity, links_fftt):
    links_output_curve = pd.DataFrame(index=np.arange(0, MINS_PER_DAY, MIN_INTERVALS))
    i = 0
    for column in links_input_curve:
        cum_arrival_curve = links_input_curve[column]
        capacity = links_capacity[i]
        fftt = links_fftt[i]
        cum_departure_curve = cum_arrival_curve.copy()
        for j in range(NUM_BINS):
            if cum_arrival_curve.iloc[j] > cum_departure_curve.iloc[j-1] + capacity * MIN_INTERVALS:
                cum_departure_curve.iloc[j] = cum_departure_curve.iloc[j - 1] + capacity * MIN_INTERVALS
        cum_departure_curve = cum_departure_curve.shift(int(fftt / MIN_INTERVALS)).fillna(0)
        links_output_curve[column] = cum_departure_curve.values
        i += 1
    return links_output_curve