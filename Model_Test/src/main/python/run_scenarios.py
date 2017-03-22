from generate_demand import generate_initial_demand
from generate_io_curves import get_links_output_curve, get_congestion_links_input_curve_after_merging,\
    get_congestion_links_input_curve_from_demand, get_freeway_links_input_curve_after_diverging
from route_choice import check_route_choice
from queue_spillover import check_queue_spillover
from calculate_link_demand_and_congestion import get_link_congestion, get_link_demand
from plot_curves import plot_io_curves, plot_demand_congestion

MINS_PER_DAY = 1440

class Parameters():

    # Scenario 1 Parameters

    min_intervals = 5
    num_bins = MINS_PER_DAY / min_intervals
    num_zones = 4
    demand_start_times = [0, 480, 960]
    demand_end_times = [240, 720, 1200]
    demand_slopes=[0.1,0.1,0.1]
    congestion_links_capacity=[10,10,10,5]
    threshold_output_for_congestion = [1,1,1,1]
    congestion_links_fftt=[20,20,20,20]
    congestion_links_jam_density = [100,100,100,100]
    congestion_links_length = [100,100,100,100]
    freeway_links_capacity=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,20, 20, 20, 20]
    freeway_links_fftt=[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    freeway_links_jam_density = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    freeway_links_length = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    congestion_nn_smoothening_number = [10,10,10,10]
    check_route_choice = False
    plot_congestion_io_curves = True
    plot_demand_congestion_curves = True
    plot_route_choice_io_curves = False
    check_queue_spillover = False
    file_directory = './scenario_1'
    get_curves_data = False

def run(parameters):
    od_demand_funcs = generate_initial_demand(num_zones=parameters.num_zones, start_times=parameters.demand_start_times,
                                              end_times=parameters.demand_end_times, slopes=parameters.demand_slopes)
    congestion_links_input_curve_from_zone = get_congestion_links_input_curve_from_demand(num_zones=parameters.num_zones,
                                                                                      od_demand_funcs=od_demand_funcs,
                                                                                min_intervals=parameters.min_intervals)
    congestion_links_output_curve_from_zone = get_links_output_curve(links_input_curve=
                                                                 congestion_links_input_curve_from_zone,
                                                                 links_capacity=parameters.congestion_links_capacity,
                                                                 links_fftt=parameters.congestion_links_fftt,
                                                                 min_intervals=parameters.min_intervals,
                                                                 num_bins=parameters.num_bins)
    freeway_links_input_curve = get_freeway_links_input_curve_after_diverging(congestion_links_output_curve=
                                                                          congestion_links_output_curve_from_zone,
                                                                          num_zones=parameters.num_zones,
                                                                          min_intervals=parameters.min_intervals,
                                                                          num_bins=parameters.num_bins)
    freeway_links_output_curve = get_links_output_curve(links_input_curve=freeway_links_input_curve,
                                                    links_capacity=parameters.freeway_links_capacity,
                                                    links_fftt=parameters.freeway_links_fftt,
                                                    min_intervals=parameters.min_intervals,
                                                    num_bins=parameters.num_bins)
    if parameters.check_route_choice:
        best_route_input_curve = freeway_links_input_curve[parameters.num_zones - 1]
        best_route_output_curve = freeway_links_output_curve[parameters.num_zones - 1]
        alternate_route = [(freeway_links_input_curve[parameters.num_zones - 2],
                              freeway_links_output_curve[parameters.num_zones - 2]),
                             (freeway_links_input_curve[parameters.num_zones * (parameters.num_zones - 1) - 1],
                              freeway_links_output_curve[parameters.num_zones * (parameters.num_zones - 1) - 1])]
        best_route_fftt = parameters.freeway_links_fftt[parameters.num_zones - 1]
        best_route_bottleneck_capacity = parameters.freeway_links_capacity[parameters.num_zones - 1]
        alternative_route_fftts = [parameters.freeway_links_fftt[parameters.num_zones - 2],
                                   parameters.freeway_links_fftt[parameters.num_zones * (parameters.num_zones - 1) - 1]]
        check_route_choice(best_route_input_curve=best_route_input_curve, best_route_output_curve=best_route_output_curve,
                           alternate_route=alternate_route,best_route_fftt=best_route_fftt,
                           best_route_bottleneck_capacity=best_route_bottleneck_capacity,
                           alternate_route_fftts=alternative_route_fftts,
                           min_intervals=parameters.min_intervals, num_bins=parameters.num_bins)

    if parameters.plot_route_choice_io_curves:
        io_series = [
            (freeway_links_input_curve.as_matrix()[:, parameters.num_zones - 1],
             freeway_links_output_curve.as_matrix()[:, parameters.num_zones - 1]),
            (freeway_links_input_curve.as_matrix()[:, parameters.num_zones - 2],
             freeway_links_output_curve.as_matrix()[:, parameters.num_zones - 2]),
            (freeway_links_input_curve.as_matrix()[:, parameters.num_zones * (parameters.num_zones - 1) - 1],
             freeway_links_output_curve.as_matrix()[:, parameters.num_zones * (parameters.num_zones - 1) - 1])
        ]
        plot_io_curves(io_series=io_series, filepath=parameters.file_directory + '/sample_plots/io_curve_route_choice_links.png',
                       min_intervals=parameters.min_intervals)

    congestion_links_input_curve_to_zone = get_congestion_links_input_curve_after_merging(freeway_links_output_curve=
                                                                                      freeway_links_output_curve,
                                                                                      num_zones=parameters.num_zones,
                                                                                min_intervals=parameters.min_intervals)
    congestion_links_output_curve_to_zone = get_links_output_curve(links_input_curve=congestion_links_input_curve_to_zone,
                                                               links_capacity=parameters.congestion_links_capacity,
                                                               links_fftt=parameters.congestion_links_fftt,
                                                               min_intervals=parameters.min_intervals,
                                                               num_bins=parameters.num_bins)

    if parameters.check_queue_spillover:
        check_queue_spillover(links_input_curve=congestion_links_input_curve_to_zone,
                              links_output_curve=congestion_links_output_curve_to_zone,
                              fftts=parameters.congestion_links_fftt,
                              links_capacity=parameters.congestion_links_capacity,
                              links_jam_density=parameters.congestion_links_jam_density,
                              links_length=parameters.congestion_links_length,
                              min_intervals=parameters.min_intervals,
                              num_bins=parameters.num_bins)
        for i in range(parameters.num_bins):
            freeway_links_output_curve[parameters.num_zones - 1].iloc[i] = min(freeway_links_output_curve[parameters.num_zones - 1].iloc[i],
                                                                    congestion_links_input_curve_to_zone[parameters.num_zones - 1].iloc[i])
        spillover_freeway_link_indices = [(parameters.num_zones * (parameters.num_zones - 1) + k) for k in range(1, parameters.num_zones)]
        spillover_freeway_link_indices = spillover_freeway_link_indices + [parameters.num_zones - 1]
        spillover_freeway_links_input_curves = freeway_links_input_curve[spillover_freeway_link_indices].copy()
        spillover_freeway_links_output_curves = freeway_links_output_curve[spillover_freeway_link_indices].copy()
        check_queue_spillover(links_input_curve=spillover_freeway_links_input_curves,
                            links_output_curve=spillover_freeway_links_output_curves,
                            fftts=parameters.freeway_links_fftt[:parameters.num_zones],
                            links_capacity= parameters.freeway_links_capacity[:parameters.num_zones],
                            links_jam_density=parameters.freeway_links_jam_density[:parameters.num_zones],
                            links_length=parameters.freeway_links_length[:parameters.num_zones],
                            min_intervals=parameters.min_intervals, num_bins=parameters.num_bins)
        for i in range(parameters.num_bins):
            freeway_links_input_curve[parameters.num_zones - 1].iloc[i] = \
                spillover_freeway_links_input_curves[parameters.num_zones - 1].iloc[i]
            congestion_links_output_curve_from_zone[0].iloc[i] = \
                spillover_freeway_links_input_curves[parameters.num_zones - 1].iloc[i]
        congestion_spillover = get_link_congestion(link_input_curve=congestion_links_input_curve_from_zone[0],
                                                   link_output_curve=congestion_links_output_curve_from_zone[0],
                                            threshold_output_for_congestion=parameters.threshold_output_for_congestion[0],
                                            congestion_nn_smoothening_number=parameters.congestion_nn_smoothening_number[0],
                                                   min_intervals=parameters.min_intervals, num_bins=parameters.num_bins)

    congestion_values = get_link_congestion(link_input_curve=congestion_links_input_curve_to_zone[parameters.num_zones - 1],
                                        link_output_curve=congestion_links_output_curve_to_zone[parameters.num_zones - 1],
                    threshold_output_for_congestion=parameters.threshold_output_for_congestion[parameters.num_zones - 1],
                    congestion_nn_smoothening_number=parameters.congestion_nn_smoothening_number[parameters.num_zones - 1],
                                                    min_intervals=parameters.min_intervals, num_bins=parameters.num_bins)
    link_demands = [get_link_demand(link_input_curve=congestion_links_input_curve_from_zone[i],
                                    num_bins=parameters.num_bins)
                for i in range(parameters.num_zones - 1)]
    io_series = [
    (congestion_links_input_curve_to_zone.as_matrix()[:, parameters.num_zones - 1],
     congestion_links_output_curve_to_zone.as_matrix()[:, parameters.num_zones - 1])]
    if parameters.plot_congestion_io_curves:
        plot_io_curves(io_series=io_series, filepath=parameters.file_directory+'/sample_plots/io_curve_congestion_zone_link.png',
                       min_intervals=parameters.min_intervals)
    if parameters.plot_demand_congestion_curves:
        plot_demand_congestion(demands=link_demands, congestion=congestion_values,
                       filepath=parameters.file_directory+'/sample_plots/demand_congestion_plot.png')
    if parameters.check_queue_spillover:
        plot_demand_congestion(demands=link_demands, congestion=congestion_values,
                               filepath=parameters.file_directory+'/sample_plots/spillover_congestion_plot.png',
                               congestion_spillover=congestion_spillover)
    if parameters.get_curves_data:
        return link_demands, congestion_values, congestion_links_input_curve_from_zone, \
               congestion_links_output_curve_from_zone, freeway_links_input_curve, freeway_links_output_curve, \
               congestion_links_input_curve_to_zone, congestion_links_output_curve_to_zone

if __name__ == '__main__':

    parameters = Parameters()

    # Scenario 1
    # 4 zones, demands from 3 zones well separated
    # No route choice
    # No congestion spillover
    run(parameters=parameters)

    # Scenario 2
    # 4 zones, demands from 3 zones with overlap between two
    # No route choice
    # No congestion spillover
    parameters.demand_start_times = [0,200,1000]
    parameters.demand_end_times = [240, 440, 1240]
    parameters.file_directory = './scenario_2'
    run(parameters=parameters)

    # Scenario 3
    # 4 zones, demands from 3 zones well separated
    # Congestion on freeway + No route choice
    # No congestion spillover
    parameters = Parameters()
    parameters.freeway_links_capacity = [20,20,20,4,20,20,20,20,20,20,20,20,20,20,20,20]
    parameters.plot_route_choice_io_curves = True
    parameters.file_directory = './scenario_3'
    run(parameters=parameters)

    # Scenario 4
    # 4 zones, demands from 3 zones well separated
    # Congestion on freeway + Route choice
    # No congestion spillover
    parameters = Parameters()
    parameters.freeway_links_capacity = [20, 20, 20, 4, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    parameters.check_route_choice = True
    parameters.plot_route_choice_io_curves = True
    parameters.file_directory = './scenario_4'
    run(parameters=parameters)

    # Scenario 5
    # 4 zones, demands from 3 zones well separated
    # No route choice
    # Plotting congestion in two zones + No congestion spillover
    parameters = Parameters()
    parameters.check_queue_spillover = True
    parameters.file_directory = './scenario_5'
    run(parameters=parameters)

    # Scenario 6
    # 4 zones, demands from 3 zones well separated
    # No route choice
    # Plotting congestion in two zones + Congestion spillover
    parameters = Parameters()
    parameters.check_queue_spillover = True
    parameters.congestion_links_jam_density = [100,100,100,25]
    parameters.congestion_links_length = [100, 100, 100, 10]
    parameters.freeway_links_jam_density = [100,100,100,25,100,100,100,100,100,100,100,100,100,100,100,100]
    parameters.freeway_links_length = [100,100,100,10,100,100,100,100,100,100,100,100,100,100,100,100]
    parameters.file_directory = './scenario_6'
    run(parameters=parameters)
