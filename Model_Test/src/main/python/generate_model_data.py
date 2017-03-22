from run_scenarios import Parameters, run
import csv
import numpy as np

np.random.seed(0)

MINS_PER_DAY = 1440
default_parameters = Parameters()
slope_noise_std = 0.01
start_time_noise_std = 30
realistic_demand_start_times = [360,420,480]
realistic_demand_congestion_link_capacities = [10,10,10,15]

def get_header(link_demands, congestion_values, congestion_links_input_curve_from_zone,
    congestion_links_output_curve_from_zone, freeway_links_input_curve, freeway_links_output_curve,
    congestion_links_input_curve_to_zone, congestion_links_output_curve_to_zone):
    header_row = ['Time(min)']
    for i in range(len(link_demands)):
        header_row.append('Link ' + str(i) + ' demand')
    header_row.append('Congestion value')
    for i in range(len(congestion_links_input_curve_from_zone.columns)):
        header_row.append('Congestion Link from zone ' + str(i) + ' input count')
    for i in range(len(congestion_links_output_curve_from_zone.columns)):
        header_row.append('Congestion Link from zone ' + str(i) + ' output count')
    for i in range(len(freeway_links_input_curve.columns)):
        header_row.append('Freeway Link ' + str(i) + ' input count')
    for i in range(len(freeway_links_output_curve.columns)):
        header_row.append('Freeway Link ' + str(i) + ' output count')
    for i in range(len(congestion_links_input_curve_to_zone.columns)):
        header_row.append('Congestion Link to zone ' + str(i) + ' input count')
    for i in range(len(congestion_links_output_curve_to_zone.columns)):
        header_row.append('Congestion Link to zone ' + str(i) + ' output count')
    return header_row

def write_csv(filename, start_day, end_day, parameters, variable_slope = False, variable_times = False,
              realistic_demand = False, min_intervals=5):
    link_demands, congestion_values, congestion_links_input_curve_from_zone, \
    congestion_links_output_curve_from_zone, freeway_links_input_curve, freeway_links_output_curve, \
    congestion_links_input_curve_to_zone, congestion_links_output_curve_to_zone = run(parameters=parameters)
    with open(filename,'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if start_day == 0:
            header_row = get_header(link_demands, congestion_values, congestion_links_input_curve_from_zone,
                                    congestion_links_output_curve_from_zone, freeway_links_input_curve,
                                    freeway_links_output_curve, congestion_links_input_curve_to_zone,
                                    congestion_links_output_curve_to_zone)
            writer.writerow(header_row)
        for i in range(start_day,end_day):
            for j in range(0, MINS_PER_DAY, min_intervals):
                row_data = [i * MINS_PER_DAY + j]
                for k in range(len(link_demands)):
                    try:
                        row_data.append(link_demands[k][j / min_intervals])
                    except:
                        row_data.append(link_demands[k][j / min_intervals - 1])
                try:
                    row_data.append(congestion_values[j / min_intervals])
                except:
                    row_data.append(congestion_values[j / min_intervals - 1])
                for k in range(len(congestion_links_input_curve_from_zone.columns)):
                    row_data.append(congestion_links_input_curve_from_zone[k][j])
                for k in range(len(congestion_links_output_curve_from_zone.columns)):
                    row_data.append(congestion_links_output_curve_from_zone[k][j])
                for k in range(len(freeway_links_input_curve.columns)):
                    row_data.append(freeway_links_input_curve[k][j])
                for k in range(len(freeway_links_output_curve.columns)):
                    row_data.append(freeway_links_output_curve[k][j])
                for k in range(len(congestion_links_input_curve_to_zone.columns)):
                    row_data.append(congestion_links_input_curve_to_zone[k][j])
                for k in range(len(congestion_links_output_curve_to_zone.columns)):
                    row_data.append(congestion_links_output_curve_to_zone[k][j])
                writer.writerow(row_data)
            if variable_slope:
                parameters.demand_slopes = default_parameters.demand_slopes + np.random.rand(3) * slope_noise_std
            if variable_times:
                parameters.demand_start_times = default_parameters.demand_start_times + np.random.randn(3) * \
                                                                                        start_time_noise_std
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
            link_demands, congestion_values, congestion_links_input_curve_from_zone, \
            congestion_links_output_curve_from_zone, freeway_links_input_curve, freeway_links_output_curve, \
            congestion_links_input_curve_to_zone, congestion_links_output_curve_to_zone = run(parameters=parameters)
    csvfile.close()

if __name__ == '__main__':

    #Scenario 1, Constant
    parameters = Parameters()
    parameters.plot_congestion_io_curves = False
    parameters.plot_demand_congestion_curves = False
    parameters.get_curves_data = True
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_constant.csv', start_day=0, end_day=1000,
              parameters=parameters,min_intervals=parameters.min_intervals)

    #Scenario 1 + Variable slope
    parameters.demand_slopes += np.random.randn(3)*slope_noise_std
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_random_volume.csv', start_day=0, end_day=1000,
              parameters=parameters,variable_slope=True,min_intervals=parameters.min_intervals)

    #Scenario 1 + Variable times
    parameters.demand_slopes = default_parameters.demand_slopes
    parameters.demand_start_times = default_parameters.demand_start_times + np.random.randn(3) * start_time_noise_std
    parameters.demand_start_times[parameters.demand_start_times < 0] = 0
    parameters.demand_end_times = parameters.demand_start_times + np.array(np.array(default_parameters.demand_end_times)
                                                                    - np.array(default_parameters.demand_start_times))
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_random_times.csv', start_day=0, end_day=1000,
              parameters=parameters,variable_times=True, min_intervals=parameters.min_intervals)

    #Scenario 1 + Variable slope + Variable times
    parameters.demand_slopes += np.random.randn(3) * slope_noise_std
    parameters.demand_start_times = default_parameters.demand_start_times + np.random.randn(3) * start_time_noise_std
    parameters.demand_start_times[parameters.demand_start_times < 0] = 0
    parameters.demand_end_times = parameters.demand_start_times + np.array(np.array(default_parameters.demand_end_times)
                                                                    - np.array(default_parameters.demand_start_times))
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_random_volume_random_times.csv', start_day=0,
              end_day=1000, parameters=parameters, variable_slope=True, variable_times=True,
              min_intervals=parameters.min_intervals)

    #Scenario 1 + Realistic demand
    parameters.congestion_links_capacity = realistic_demand_congestion_link_capacities
    parameters.demand_slopes = parameters.demand_slopes + np.random.rand(3) * 0.01
    parameters.demand_start_times = np.array(realistic_demand_start_times) + np.random.randn(3) * 60
    parameters.demand_end_times = parameters.demand_start_times + np.array(np.array(default_parameters.demand_end_times)
                                                                    - np.array(default_parameters.demand_start_times))
    parameters.threshold_output_for_congestion = [1, 1, 1, 1]
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_realistic.csv',
              start_day=0,end_day=1000,parameters=parameters, variable_slope=True, variable_times=True,
              min_intervals=parameters.min_intervals)
