import numpy as np

MINS_PER_DAY = 1440

def get_link_demand(link_input_curve,num_bins):
    link_demand = []
    for i in range(1, num_bins):
        link_demand.append(link_input_curve.iloc[i] - link_input_curve.iloc[i-1])
    return link_demand

def get_link_congestion(link_input_curve, link_output_curve, threshold_output_for_congestion,
                        congestion_nn_smoothening_number,min_intervals, num_bins):
    congestion = []
    for i in range(1,num_bins):
        output = (link_output_curve.iloc[i] - link_output_curve.iloc[i-1])/float(min_intervals)
        accumulation = link_input_curve.iloc[i] - link_output_curve.iloc[i]
        congestion.append(accumulation/output if output > threshold_output_for_congestion else 0)
    for i in range(len(congestion)):
        congestion[i] = np.average(congestion[max(i-congestion_nn_smoothening_number,0):
                        min(i+congestion_nn_smoothening_number+1,len(congestion)+1)])
    return congestion