MINS_PER_DAY = 1440

def check_queue_spillover(links_input_curve, links_output_curve, fftts,links_capacity, links_jam_density, links_length,
                          min_intervals,num_bins):
    for i in range(len(links_input_curve.columns)):
        column_name = links_input_curve.columns[i]
        link_input_curve = links_input_curve[column_name].copy()
        link_output_curve = links_output_curve[column_name].copy()
        fftt = fftts[i]
        link_capacity = links_capacity[i]
        link_jam_density = links_jam_density[i]
        link_length = links_length[i]
        link_cum_departure_shifted = link_output_curve.copy().shift(-int(fftt / min_intervals)).fillna(method='ffill')
        link_cum_departure_transformed = link_cum_departure_shifted.copy()\
            .shift(int(link_jam_density/float(link_capacity))).fillna(0)\
            .apply(lambda x: x + link_jam_density*link_length)
        if (link_cum_departure_transformed < link_input_curve).any():
            print "Queue spillover"
            for j in range(num_bins):
                link_input_curve.iloc[j] = min(link_input_curve.iloc[j], link_cum_departure_transformed.iloc[j])
                links_input_curve[column_name].iloc[j] = link_input_curve.iloc[j]
        else:
            print "No queue spillover"
    return links_input_curve, links_output_curve