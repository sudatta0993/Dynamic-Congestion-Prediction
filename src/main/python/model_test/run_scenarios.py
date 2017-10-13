from generate_demand import generate_initial_demand
from generate_io_curves import get_links_output_curve, get_congestion_links_input_curve_after_merging,\
    get_congestion_links_input_curve_from_demand, get_freeway_links_input_curve_after_diverging, \
    get_congestion_links_input_curve_after_toll
from route_choice import check_route_choice
from queue_spillover import check_queue_spillover
from calculate_link_demand_and_congestion import get_link_congestion, \
    get_link_demand, get_congestion_marginal_impact, get_cumulative_congestion_value
from plot_curves import plot_io_curves, plot_demand_congestion
import numpy as np
import os

MINS_PER_DAY = 1440

np.random.seed(0)

class Parameters():

    # Initialize with dictionary or default parameters (Scenario 1)
    def __init__(self,dict=None):
        if not dict:
            dict = {}
        self.min_intervals = dict.get('min_intervals',5)
        self.num_bins = MINS_PER_DAY / self.min_intervals
        self.num_zones = dict.get('num_zones',4)
        self.demand_start_times = dict.get('demand_start_times',[0, 480, 960])
        self.demand_end_times = dict.get('demand_end_times',[240, 720, 1200])
        self.demand_slopes = dict.get('demand_slopes',[0.1,0.1,0.1])
        self.congestion_links_capacity = dict.get('congestion_links_capacity',[10,10,10,5])
        self.threshold_output_for_congestion = dict.get('threshold_output_for_congestion',[1,1,1,1])
        self.threshold_beta_for_congestion_impact = dict.get('threshold_output_for_congestion_impact',[0.01,0.01,0.01,0.01])
        self.congestion_links_fftt = dict.get('congestion_links_fftt',[20,20,20,20])
        self.congestion_links_jam_density = dict.get('congestion_links_jam_density',[100,100,100,100])
        self.congestion_links_length = dict.get('congestion_links_length',[100,100,100,100])
        self.freeway_links_capacity = dict.get('freeway_links_capacity',[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,20, 20, 20, 20])
        self.freeway_links_fftt = dict.get('freeway_links_fftt',[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])
        self.freeway_links_jam_density = dict.get('freeway_links_jam_density',[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])
        self.freeway_links_length = dict.get('freeway_links_length',[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])
        self.congestion_nn_smoothening_number = dict.get('congestion_nn_smoothening_number',[10,10,10,10])
        self.congestion_marginal_impact_nn_smoothening_number = dict.get('congestion_marginal_impact_nn_smoothening_number',[10,10,10,10])
        self.check_route_choice = dict.get('check_route_choice',False)
        self.plot_congestion_io_curves = dict.get('plot_congestion_io_curves',True)
        self.plot_demand_congestion_curves = dict.get('plot_demand_congestion_curves',True)
        self.plot_route_choice_io_curves = dict.get('plot_route_choice_io_curves',False)
        self.check_queue_spillover = dict.get('check_queue_spillover',False)
        self.file_directory = dict.get('file_directory','./scenario_1')
        self.get_curves_data = dict.get('get_curves_data',False)
        self.incident_prob = dict.get('incident_prob',0)
        self.incident_time = dict.get('incident_time',int(np.random.rand()*MINS_PER_DAY))
        self.plot_demand_congestion_marginal_impact_curves = dict.get('plot_demand_congestion_marginal_impact_curves',True)
        self.value_of_time_early = dict.get('value_of_time_early',[0.5,0.5,0.5])
        self.value_of_time_late = dict.get('value_of_time_late',[0.5,0.5,0.5])
        self.implement_toll = dict.get('implement_toll',False)
        self.toll_curves = dict.get('toll_curves',[lambda x: 0,lambda x: 0,lambda x: 0])
        self.plot_cum_input_curves_toll = dict.get('plot_cum_input_curves_toll',False)
        self.demand_nn_smoothening_number = dict.get('demand_nn_smoothening_number',[20,20,20])

def run(parameters):
    od_demand_funcs = generate_initial_demand(num_zones=parameters.num_zones, start_times=parameters.demand_start_times,
                                              end_times=parameters.demand_end_times, slopes=parameters.demand_slopes)
    congestion_links_input_curve_from_zone = get_congestion_links_input_curve_from_demand(num_zones=parameters.num_zones,
                                                                                      od_demand_funcs=od_demand_funcs,
                                                                                min_intervals=parameters.min_intervals)
    if parameters.implement_toll:
        congestion_links_input_curve_from_zone = \
            get_congestion_links_input_curve_after_toll(congestion_links_input_curve_from_demand=congestion_links_input_curve_from_zone,
                                                        toll_curves=parameters.toll_curves, min_intervals=parameters.min_intervals,
                                                        value_of_time_early=parameters.value_of_time_early,
                                                        value_of_time_late=parameters.value_of_time_late, num_zones=parameters.num_zones,
                                                        plot_cum_input_curves_toll=parameters.plot_cum_input_curves_toll,
                                                        file_directory=parameters.file_directory)
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
        incident_prob = parameters.incident_prob
        incident_occurance = np.random.rand() < incident_prob
        incident_bin = parameters.incident_time / parameters.min_intervals if incident_occurance else MINS_PER_DAY / parameters.min_intervals
        check_route_choice(best_route_input_curve=best_route_input_curve, best_route_output_curve=best_route_output_curve,
                           alternate_route=alternate_route,best_route_fftt=best_route_fftt,
                           best_route_bottleneck_capacity=best_route_bottleneck_capacity,
                           alternate_route_fftts=alternative_route_fftts,
                           min_intervals=parameters.min_intervals, num_bins=parameters.num_bins, incident_bin=incident_bin)

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
                                    num_bins=parameters.num_bins, implement_tolls=parameters.implement_toll,
                                    demand_nn_smoothening_number=parameters.demand_nn_smoothening_number[i])
                    for i in range(parameters.num_zones - 1)]
    congestion_marginal_impacts = get_congestion_marginal_impact(link_input_curve=congestion_links_input_curve_to_zone[parameters.num_zones - 1],
                                                                 link_output_curve=congestion_links_output_curve_to_zone[parameters.num_zones - 1],
                                                                 congestion_values=congestion_values,
                                                                 marginal_impact_nn_smoothening_number=parameters.
                                                                 congestion_marginal_impact_nn_smoothening_number[parameters.num_zones - 1],
                                                                 min_intervals=parameters.min_intervals, num_bins=parameters.num_bins,
                                                                 threshold_beta_for_congestion_impact=
                                                                 parameters.threshold_beta_for_congestion_impact[parameters.num_zones - 1])
    cum_congestion = get_cumulative_congestion_value(congestion_values,parameters.min_intervals,parameters.num_bins - 1)
    io_series = [
    (congestion_links_input_curve_to_zone.as_matrix()[:, parameters.num_zones - 1],
     congestion_links_output_curve_to_zone.as_matrix()[:, parameters.num_zones - 1])]
    if not os.path.exists(parameters.file_directory):
        os.makedirs(parameters.file_directory)
    if parameters.plot_congestion_io_curves:
        plot_io_curves(io_series=io_series, filepath=parameters.file_directory+'/sample_plots/io_curve_congestion_zone_link.png',
                       min_intervals=parameters.min_intervals)
    if parameters.plot_demand_congestion_curves:
        plot_demand_congestion(demands=link_demands, congestion=congestion_values,
                               filepath=parameters.file_directory+'/sample_plots/demand_congestion_plot.png',
                               num_bins=parameters.num_bins, min_intervals=parameters.min_intervals)
    if parameters.check_queue_spillover:
        plot_demand_congestion(demands=link_demands, congestion=congestion_values,
                               filepath=parameters.file_directory+'/sample_plots/spillover_congestion_plot.png',
                               congestion_spillover=congestion_spillover,
                               num_bins=parameters.num_bins, min_intervals=parameters.min_intervals)
    if parameters.plot_demand_congestion_marginal_impact_curves:
        plot_demand_congestion(demands=[link_demand[:-1] for link_demand in link_demands], congestion=congestion_marginal_impacts,
                               filepath=parameters.file_directory + '/sample_plots/demand_marginal_impact_plot.png',
                               num_bins=parameters.num_bins - 1, min_intervals=parameters.min_intervals)
    if parameters.get_curves_data:
        dict_return = {'link_demands':link_demands, 'congestion_values':congestion_values,
                       'congestion_marginal_impact_values': congestion_marginal_impacts,
                       'congestion_links_input_curve_from_zone':congestion_links_input_curve_from_zone,
                       'congestion_links_output_curve_from_zone':congestion_links_output_curve_from_zone,
                       'freeway_links_input_curve':freeway_links_input_curve,
                       'freeway_links_output_curve':freeway_links_output_curve,
                       'congestion_links_input_curve_to_zone':congestion_links_input_curve_to_zone,
                       'congestion_links_output_curve_to_zone': congestion_links_output_curve_to_zone,
                       'cum_congestion': cum_congestion}
        if parameters.check_queue_spillover:
            dict_return['congestion_spillover'] = congestion_spillover
        return dict_return
