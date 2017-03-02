def no_demand():
    return lambda x: 0

def simple_h_w_demand_func(start_time, end_time, slope):
    half_interval = (end_time - start_time)/2
    peak_time = start_time + half_interval
    peak = slope*half_interval
    return lambda x: 0 if x < start_time else slope*(x - start_time) if x < peak_time \
        else (peak - slope*(x-peak_time)) if x < end_time else 0

def generate_initial_demand(num_zones, start_times, end_times, slopes):
    od_demand_funcs = []
    for i in range(num_zones):
        for j in range(num_zones):
            if j == num_zones - 1 and j!= i:
                od_demand_funcs.append(simple_h_w_demand_func(start_time=start_times[i],
                                                end_time=end_times[i], slope=slopes[i]))
            else:
                od_demand_funcs.append(no_demand())
    return od_demand_funcs