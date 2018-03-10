import unittest
from src.main.python.model_test.run_scenarios import Parameters, run
import numpy as np
import os

class test_run_scenarios(unittest.TestCase):
    def test_scenario_no_toll(self):
        parameters = Parameters()
        run(parameters=parameters)
        parameters.file_directory = './scenario_no_toll'
        parameters.demand_start_times = [50, 465, 959]
        parameters.demand_end_times = [290, 705, 1199]
        parameters.demand_slopes = [0.1141565623, 0.1098967944, 0.1047348826]
        parameters.get_curves_data = True
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        print "Cumulative congestion value for No Toll = " + str(cum_congestion)
        print "Total toll collected for No Toll = " + str(0)


    def test_scenario_optimal_toll(self):
        parameters = Parameters()
        run(parameters=parameters)
        parameters.file_directory = './scenario_optimal_toll'
        parameters.demand_start_times = [50, 465, 959]
        parameters.demand_end_times = [290, 705, 1199]
        parameters.demand_slopes = [0.1141565623, 0.1098967944, 0.1047348826]
        parameters.get_curves_data = True
        parameters.toll_curves = [lambda x: 122.5 + x * 0.5 if x < 50 else 147.5 - x * 0.5 if x < 345 else 0,
                                  lambda x: 0 if x < 345 else (x - 345) / 2.0 if x < 585 else 120 - (x - 585) / 2.0 if x < 840 else 0,
                                  lambda x: 0 if x < 720 else (x - 720) / 2.0 if x < 1200 else 240]
        parameters.implement_toll = True
        parameters.plot_cum_input_curves_toll = True
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        total_toll_collected = dict['total_toll_collected']
        print "Cumulative congestion value for Optimal Toll = " + str(cum_congestion)
        print "Total toll collected for Optimal Toll = " + str(sum(total_toll_collected))


    def test_scenario_average_toll(self):
        parameters = Parameters()
        parameters.num_bins = 1440
        parameters.min_intervals = 1
        parameters.demand_start_times = [50, 465, 959]
        parameters.demand_end_times = [290, 705, 1199]
        parameters.demand_slopes = [0.1141565623, 0.1098967944, 0.1047348826]
        parameters.plot_cum_input_curves_toll = True
        parameters.implement_toll = True
        parameters.file_directory = './scenario_average_toll'
        parameters.get_curves_data = True
        parameters.toll_curves = [lambda x: 240 - x * 0.5 if x < 480 else 0,
                                  lambda x: 0 if x < 360 else (x - 360) / 2.0 if x < 600 else 120 - (x - 600) / 2.0 if x < 840 else 0,
                                  lambda x: 0 if x < 720 else (x - 720) / 2.0 if x < 1200 else 240]
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        total_toll_collected = dict['total_toll_collected']
        print "Cumulative congestion value for Average Toll = " + str(cum_congestion)
        print "Total toll collected for Average Toll = " + str(sum(total_toll_collected))


    def test_scenario_model_toll(self):
        parameters = Parameters()
        parameters.num_bins = 1440
        parameters.min_intervals = 1
        parameters.demand_start_times = [50, 465, 959]
        parameters.demand_end_times = [290, 705, 1199]
        parameters.demand_slopes = [0.1141565623, 0.1098967944, 0.1047348826]
        parameters.plot_cum_input_curves_toll = True
        parameters.implement_toll = True
        parameters.file_directory = './scenario_model_toll'
        parameters.get_curves_data = True
        toll_values = np.zeros((288, 6))
        for i in range(12):
            current_directory = os.path.dirname(__file__)
            for j in range(3):
                parent_directory = os.path.split(current_directory)[0]
                current_directory = parent_directory
            start_time = ((i + 2) * 120) % 1440
            end_time = ((i + 3) * 120) % 1440 if ((i + 3) * 120) % 1440 != 0 else 1440
            filepath = os.path.join(parent_directory,
                                    'main/python/model_test/scenario_1/toll/attention_st_' + str(start_time) + '_et_' + str(
                                        end_time) + '.csv')
            attention_values = np.genfromtxt(filepath, delimiter=',')
            toll_values[i * 24:i * 24 + 23, :] = 91 * attention_values
            if (i < 11):
                toll_values[(i + 1) * 24, :] = toll_values[(i + 1) * 24 - 1, :]
        parameters.toll_curves = [lambda x: toll_values[int(x / 5), 0] + toll_values[int(x / 5), 3],
                                  lambda x: toll_values[int(x / 5), 1] + toll_values[int(x / 5), 4],
                                  lambda x: toll_values[int(x / 5), 2] + toll_values[int(x / 5), 5]]
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        total_toll_collected = dict['total_toll_collected']
        print "Cumulative congestion value for Model-based Toll = " + str(cum_congestion)
        print "Total toll collected for Model-based Toll = " + str(sum(total_toll_collected))


    def test_scenario_nn_toll(self):
        parameters = Parameters()
        run(parameters=parameters)
        parameters.file_directory = './scenario_nn_toll'
        parameters.demand_start_times = [50, 465, 959]
        parameters.demand_end_times = [290, 705, 1199]
        parameters.demand_slopes = [0.1141565623, 0.1098967944, 0.1047348826]
        # Last day
        # parameters.demand_start_times = [16, 503, 917]
        # parameters.demand_end_times = [256, 743, 1157]
        # parameters.demand_slopes = [0.1015559073, 0.1077033088, 0.1162790698]
        parameters.get_curves_data = True
        parameters.toll_curves = [lambda x: 175.5 + x * 0.5 if x < 16 else 183.5 - x * 0.5 if x < 383 else 0,
                                  lambda x: 0 if x < 383 else (x - 383) / 2.0 if x < 623 else 120 - (x - 623) / 2.0 if x < 863 else 0,
                                  lambda x: 0 if x < 677 else (x - 677) / 2.0 if x < 1157 else 240]

        parameters.implement_toll = True
        parameters.plot_cum_input_curves_toll = True
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        total_toll_collected = dict['total_toll_collected']
        print "Cumulative congestion value for 1NN-based Toll = " + str(cum_congestion)
        print "Total toll collected for 1NN-based Toll = " + str(sum(total_toll_collected))