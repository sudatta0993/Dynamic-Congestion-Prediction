import numpy as np

MINS_PER_DAY = 1440

def nn_smoothen(curve,nn_smoothening_number):
    for i in range(len(curve)):
        curve[i] = np.average(curve[max(i-nn_smoothening_number,0):
                        min(i+nn_smoothening_number+1,len(curve)+1)])
    return curve

def get_link_demand(link_input_curve,num_bins,implement_tolls, demand_nn_smoothening_number):
    link_demand = []
    for i in range(1, num_bins):
        link_demand.append(link_input_curve.iloc[i] - link_input_curve.iloc[i-1])
    if implement_tolls:
        link_demand = nn_smoothen(link_demand, demand_nn_smoothening_number)
    return link_demand

def get_link_congestion(link_input_curve, link_output_curve, threshold_output_for_congestion,
                        congestion_nn_smoothening_number,min_intervals, num_bins):
    congestion = []
    for i in range(1,num_bins):
        output = (link_output_curve.iloc[i] - link_output_curve.iloc[i-1])/float(min_intervals)
        accumulation = link_input_curve.iloc[i] - link_output_curve.iloc[i]
        congestion.append(accumulation/output if output > threshold_output_for_congestion else 0)
    congestion = nn_smoothen(congestion, congestion_nn_smoothening_number)
    return congestion

def get_congestion_marginal_impact(link_input_curve, link_output_curve, congestion_values,threshold_beta_for_congestion_impact,
                                   marginal_impact_nn_smoothening_number,min_intervals, num_bins):
    marginal_impact = []
    for i in range(1,num_bins-1):
        output = (link_output_curve.iloc[i] - link_output_curve.iloc[i - 1]) / float(min_intervals)
        delta_accumulation = (link_input_curve.iloc[i] - link_output_curve.iloc[i]) - \
                             (link_input_curve.iloc[i-1] - link_output_curve.iloc[i-1])
        delta_congestion = congestion_values[i] - congestion_values[i-1]
        beta_congestion = delta_congestion / float(min_intervals)
        beta_accumulation = delta_accumulation / float(min_intervals)
        marginal_impact.append(beta_congestion/(beta_accumulation+output) if
                               abs(beta_accumulation + output) > threshold_beta_for_congestion_impact else 0.0)
    marginal_impact = nn_smoothen(nn_smoothen(marginal_impact, marginal_impact_nn_smoothening_number), marginal_impact_nn_smoothening_number)
    return marginal_impact

def get_cumulative_congestion_value(congestion_values,min_intervals,num_bins):
    cum_congestion = 0
    for i in range(num_bins - 1):
        cum_congestion += (congestion_values[i+1] + congestion_values[i]) * min_intervals / 2.0
    return cum_congestion