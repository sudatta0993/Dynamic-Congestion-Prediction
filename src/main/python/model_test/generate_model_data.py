import sys
from run_scenarios import Parameters, run
import csv
import numpy as np
import json

np.random.seed(0)

MINS_PER_DAY = 1440
default_parameters = Parameters()
slope_rho = 0
start_time_rho = 0
slope_noise_std = 0.01
start_time_noise_std = 30
realistic_demand_start_times = [360,420,480]
realistic_demand_congestion_link_capacities = [10,10,10,15]

def get_header(dict_return):
    header_row = ['Time(min)']
    for i in range(len(dict_return['link_demands'])):
        header_row.append('Link ' + str(i) + ' demand')
    header_row.append('Congestion value')
    if 'congestion_spillover' in dict_return:
        header_row.append('Congestion value in zone 0')
    for i in range(len(dict_return['congestion_links_input_curve_from_zone'].columns)):
        header_row.append('Congestion Link from zone ' + str(i) + ' input count')
    for i in range(len(dict_return['congestion_links_output_curve_from_zone'].columns)):
        header_row.append('Congestion Link from zone ' + str(i) + ' output count')
    for i in range(len(dict_return['freeway_links_input_curve'].columns)):
        header_row.append('Freeway Link ' + str(i) + ' input count')
    for i in range(len(dict_return['freeway_links_output_curve'].columns)):
        header_row.append('Freeway Link ' + str(i) + ' output count')
    for i in range(len(dict_return['congestion_links_input_curve_to_zone'].columns)):
        header_row.append('Congestion Link to zone ' + str(i) + ' input count')
    for i in range(len(dict_return['congestion_links_output_curve_to_zone'].columns)):
        header_row.append('Congestion Link to zone ' + str(i) + ' output count')
    header_row.append('Congestion marginal impact')
    return header_row

def write_csv(filename, start_day, end_day, parameters, all_slopes, all_start_times,
              variable_slope = False, variable_times = False,
              realistic_demand = False, min_intervals=5):
    dict_return = run(parameters=parameters)
    with open(filename,'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if start_day == 0:
            header_row = get_header(dict_return)
            writer.writerow(header_row)
        for i in range(start_day,end_day+1):
            for j in range(0, MINS_PER_DAY, min_intervals):
                row_data = [i * MINS_PER_DAY + j]
                for k in range(len(dict_return['link_demands'])):
                    try:
                        row_data.append(dict_return['link_demands'][k][j / min_intervals])
                    except:
                        row_data.append(dict_return['link_demands'][k][j / min_intervals - 1])
                try:
                    row_data.append(dict_return['congestion_values'][j / min_intervals])
                except:
                    row_data.append(dict_return['congestion_values'][j / min_intervals - 1])
                if parameters.check_queue_spillover:
                    try:
                        row_data.append(dict_return['congestion_spillover'][j / min_intervals])
                    except:
                        row_data.append(dict_return['congestion_spillover'][j / min_intervals - 1])
                for k in range(len(dict_return['congestion_links_input_curve_from_zone'].columns)):
                    row_data.append(dict_return['congestion_links_input_curve_from_zone'][k][j])
                for k in range(len(dict_return['congestion_links_output_curve_from_zone'].columns)):
                    row_data.append(dict_return['congestion_links_output_curve_from_zone'][k][j])
                for k in range(len(dict_return['freeway_links_input_curve'].columns)):
                    row_data.append(dict_return['freeway_links_input_curve'][k][j])
                for k in range(len(dict_return['freeway_links_output_curve'].columns)):
                    row_data.append(dict_return['freeway_links_output_curve'][k][j])
                for k in range(len(dict_return['congestion_links_input_curve_to_zone'].columns)):
                    row_data.append(dict_return['congestion_links_input_curve_to_zone'][k][j])
                for k in range(len(dict_return['congestion_links_output_curve_to_zone'].columns)):
                    row_data.append(dict_return['congestion_links_output_curve_to_zone'][k][j])
                try:
                    row_data.append(dict_return['congestion_marginal_impact_values'][j / min_intervals])
                except:
                    row_data.append(dict_return['congestion_marginal_impact_values'][j / min_intervals - 2])
                writer.writerow(row_data)
            if variable_slope:
                parameters.demand_slopes = default_parameters.demand_slopes - np.random.rand(3) * slope_noise_std
            parameters.demand_slopes = all_slopes[i - start_day]
            if variable_times:
                parameters.demand_start_times = default_parameters.demand_start_times + np.random.randn(3) * \
                                                                                        start_time_noise_std
                parameters.demand_start_times[parameters.demand_start_times < 0] = 0
                parameters.demand_end_times = parameters.demand_start_times + np.array(
                    np.array(default_parameters.demand_end_times) - np.array(default_parameters.demand_start_times))
            parameters.demand_start_times = all_start_times[i - start_day]
            parameters.demand_start_times[parameters.demand_start_times < 0] = 0
            parameters.demand_end_times = parameters.demand_start_times + np.array(
                    np.array(default_parameters.demand_end_times) - np.array(default_parameters.demand_start_times))
            if realistic_demand:
                parameters.congestion_links_capacity = realistic_demand_congestion_link_capacities
                parameters.demand_start_times = realistic_demand_start_times + np.random.randn(3) * 60
                parameters.demand_start_times[parameters.demand_start_times < 0] = 0
                parameters.demand_end_times = parameters.demand_start_times + np.array(
                    np.array(default_parameters.demand_end_times)
                    - np.array(default_parameters.demand_start_times))
            parameters.incident_time = int(np.random.rand()*MINS_PER_DAY)
            dict_return = run(parameters=parameters)
    csvfile.close()

if __name__ == '__main__':

    config_file_path = sys.argv[1]
    data_file_path = sys.argv[2]

    ### DEPRECATED (simply set slope_rho = 0, slope_noise_std = 0.01)
    variable_slope = sys.argv[3] in ["True","true","T", "t"]
    ### DEPRECATED (simply set start_time_rho = 0, start_time_noise_std = 30)
    variable_times = sys.argv[4] in ["True","true","T", "t"]

    realistic_demand = sys.argv[5] in ["True","true","T", "t"]
    start_day = int(sys.argv[6])
    end_day = int(sys.argv[7])

    with open(config_file_path) as json_file:
        dict = json.load(json_file)
        parameters = Parameters(dict)
        all_start_times = np.tile(default_parameters.demand_start_times, (end_day - start_day + 2,1))
        all_slopes = np.tile(default_parameters.demand_slopes, (end_day - start_day + 2, 1))
        for i in range(start_day+1,end_day+2):
            for j in range(len(all_start_times[i])):
                all_start_times[i,j] = default_parameters.demand_start_times[j] * (1-start_time_rho) + \
                                       all_start_times[i - 1,j]*start_time_rho + np.random.randn()*start_time_noise_std
                all_slopes[i, j] = default_parameters.demand_slopes[j] * (1 - slope_rho) + \
                                        all_slopes[i - 1, j] * slope_rho + np.random.randn() * slope_noise_std
        write_csv(filename=parameters.file_directory + '/' + data_file_path, start_day=start_day,
                  end_day=end_day, parameters=parameters, min_intervals=parameters.min_intervals,
                  variable_slope=variable_slope, variable_times=variable_times,realistic_demand=realistic_demand,
                  all_start_times=all_start_times, all_slopes=all_slopes)
    json_file.close()