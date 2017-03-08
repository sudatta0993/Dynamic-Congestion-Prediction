from run_scenarios import Parameters, run
import csv
import numpy as np

np.random.seed(0)

MINS_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MINS_PER_DAY / MIN_INTERVALS
default_parameters = Parameters()
slope_noise_std = 0.01
start_time_noise_std = 30

def write_csv(filename, num_days, parameters, variable_slope = False, variable_times = False):
    link_demands, congestion_values, congestion_links_input_curve_from_zone, \
    congestion_links_output_curve_from_zone, freeway_links_input_curve, freeway_links_output_curve, \
    congestion_links_input_curve_to_zone, congestion_links_output_curve_to_zone = run(parameters=parameters)
    with open(filename,'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header_row = ['Time(min)']
        for i in range(len(link_demands)):
            header_row.append('Link '+str(i)+' demand')
        header_row.append('Congestion value')
        for i in range(len(congestion_links_input_curve_from_zone.columns)):
            header_row.append('Congestion Link from zone ' + str(i) + ' input count')
        for i in range(len(congestion_links_output_curve_from_zone.columns)):
            header_row.append('Congestion Input from zone ' + str(i) + ' output count')
        for i in range(len(freeway_links_input_curve.columns)):
            header_row.append('Freeway Link ' + str(i) + ' input count')
        for i in range(len(freeway_links_output_curve.columns)):
            header_row.append('Freeway Link ' + str(i) + ' output count')
        for i in range(len(congestion_links_input_curve_to_zone.columns)):
            header_row.append('Congestion Link to zone ' + str(i) + ' input count')
        for i in range(len(congestion_links_output_curve_to_zone.columns)):
            header_row.append('Congestion Link to zone ' + str(i) + ' output count')
        writer.writerow(header_row)
        for i in range(num_days):
            for j in range(0, MINS_PER_DAY, MIN_INTERVALS):
                row_data = [i * MINS_PER_DAY + j]
                for k in range(len(link_demands)):
                    try:
                        row_data.append(link_demands[k][j / MIN_INTERVALS])
                    except:
                        row_data.append(link_demands[k][j / MIN_INTERVALS - 1])
                try:
                    row_data.append(congestion_values[j / MIN_INTERVALS])
                except:
                    row_data.append(congestion_values[j / MIN_INTERVALS - 1])
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
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_constant.csv', num_days=1000,
              parameters=parameters)

    #Scenario 1 + Variable slope
    parameters.demand_slopes += np.random.randn(3)*slope_noise_std
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_random_volume.csv', num_days=1000,
              parameters=parameters,variable_slope=True)

    #Scenario 1 + Variable times
    parameters.demand_slopes = default_parameters.demand_slopes
    parameters.demand_start_times = default_parameters.demand_start_times + np.random.randn(3) * start_time_noise_std
    parameters.demand_start_times[parameters.demand_start_times < 0] = 0
    parameters.demand_end_times = parameters.demand_start_times + np.array(np.array(default_parameters.demand_end_times)
                                                                    - np.array(default_parameters.demand_start_times))
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_random_times.csv', num_days=1000,
              parameters=parameters,variable_times=True)

    #Scenario 1 + Variable slope + Variable times
    parameters.demand_slopes += np.random.randn(3) * slope_noise_std
    parameters.demand_start_times = default_parameters.demand_start_times + np.random.randn(3) * start_time_noise_std
    parameters.demand_start_times[parameters.demand_start_times < 0] = 0
    parameters.demand_end_times = parameters.demand_start_times + np.array(np.array(default_parameters.demand_end_times)
                                                                    - np.array(default_parameters.demand_start_times))
    write_csv(filename=parameters.file_directory + '/lstm_input_data/input_data_random_volume_random_times.csv', num_days=1000,
              parameters=parameters, variable_slope=True, variable_times=True)