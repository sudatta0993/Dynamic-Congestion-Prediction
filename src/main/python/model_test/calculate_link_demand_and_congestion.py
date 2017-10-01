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
    for i in range(len(marginal_impact)):
        marginal_impact[i] = np.average(marginal_impact[max(i - marginal_impact_nn_smoothening_number, 0):
        min(i + marginal_impact_nn_smoothening_number + 1, len(marginal_impact) + 1)])
    for i in range(len(marginal_impact)):
        marginal_impact[i] = np.average(marginal_impact[max(i - marginal_impact_nn_smoothening_number, 0):
        min(i + marginal_impact_nn_smoothening_number + 1, len(marginal_impact) + 1)])
    return marginal_impact