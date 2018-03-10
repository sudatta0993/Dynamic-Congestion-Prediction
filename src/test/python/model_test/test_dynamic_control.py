import unittest
from src.main.python.model_test.run_scenarios import Parameters, run
import matplotlib.pyplot as plt

class test_run_scenarios(unittest.TestCase):
    '''
        Scenario 1
        4 zones, demands from 3 zones well separated
        No route choice
        No congestion spillover
    '''

    def test_scenario_1(self):
        parameters = Parameters()
        run(parameters=parameters)
        parameters.get_curves_data = True
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        print "Cumulative congestion value for Scenario 1 (No Toll) = " + str(cum_congestion)

    '''
        Scenario 8
        4 zones, demands from 3 zones well separated
        No route choice
        No congestion spillover
        Toll:
            Zone 1: Linearly decreasing (intended to delay all)
            Zone 2: Triangular (intended to prepone half and delay half)
            Zone 3: Linearly increasing (intended to prepone all)
    '''

    def test_scenario_8(self):
        parameters = Parameters()
        parameters.plot_cum_input_curves_toll = True
        parameters.implement_toll = True
        parameters.file_directory = './scenario_8'
        parameters.get_curves_data = True
        # Toll curve for Maximum Spread Out + Minimum Overlap
        parameters.toll_curves = [lambda x: 240 - x * 0.5 if x < 480 else 0,
                                  lambda x: 0 if x < 360 else (x - 360) / 2.0 if x < 600 else 120 - (x - 600) / 2.0 if x < 840 else 0,
                                  lambda x: 0 if x < 720 else (x - 720) / 2.0 if x < 1200 else 240]
        # Toll curve for Maximum Spread Out
        #parameters.toll_curves = [lambda x: 28.8*25 - x*0.5,
        #                          lambda x: 5.76*25 + x*0.5 if x < 576 else 17.28*25 - (x - 576)*0.5,
        #                          lambda x: 0.5*x if x < 1152 else 23.04*25 - (x - 1152)*0.5]
        dict = run(parameters)
        cum_congestion = dict['cum_congestion']
        total_toll_collected = dict['total_toll_collected']
        print "Cumulative congestion value for Scenario 8 (With Toll) = " + str(cum_congestion)
        print "Total toll collected = " + str(sum(total_toll_collected))
