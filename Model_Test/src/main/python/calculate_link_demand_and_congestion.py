MINS_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MINS_PER_DAY / MIN_INTERVALS

def get_link_demand(link_input_curve):
    link_demand = []
    for i in range(1, NUM_BINS):
        link_demand.append(link_input_curve.iloc[i] - link_input_curve.iloc[i-1])
    return link_demand

def get_link_congestion(link_input_curve, link_output_curve):
    congestion = []
    for i in range(1,NUM_BINS):
        output = (link_output_curve.iloc[i] - link_output_curve.iloc[i-1])/float(MIN_INTERVALS)
        accumulation = link_input_curve.iloc[i] - link_output_curve.iloc[i]
        congestion.append(accumulation/output if output > 0 else 0)
    return congestion