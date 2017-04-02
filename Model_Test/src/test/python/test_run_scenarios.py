import unittest
from src.main.python.run_scenarios import Parameters, run

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

    '''
    Scenario 2
    4 zones, demands from 3 zones with overlap between two
    No route choice
    No congestion spillover
    '''
    def test_scenario_2(self):
        parameters = Parameters()
        parameters.demand_start_times = [0, 200, 1000]
        parameters.demand_end_times = [240, 440, 1240]
        parameters.file_directory = './scenario_2'
        run(parameters=parameters)

    '''
    Scenario 3
    4 zones, demands from 3 zones well separated
    Congestion on freeway + No route choice
    No congestion spillover
    '''
    def test_scenario_3(self):
        parameters = Parameters()
        parameters.freeway_links_capacity = [20, 20, 20, 4, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        parameters.plot_route_choice_io_curves = True
        parameters.file_directory = './scenario_3'
        run(parameters=parameters)

    '''
    Scenario 4
    4 zones, demands from 3 zones well separated
    Congestion on freeway + Route choice
    No congestion spillover
    '''
    def test_scenario_4(self):
        parameters = Parameters()
        parameters.freeway_links_capacity = [20, 20, 20, 4, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        parameters.check_route_choice = True
        parameters.plot_route_choice_io_curves = True
        parameters.file_directory = './scenario_4'
        run(parameters=parameters)

    '''
    Scenario 5
    4 zones, demands from 3 zones well separated
    No route choice
    Plotting congestion in two zones + No congestion spillover
    '''
    def test_scenario_5(self):
        parameters = Parameters()
        parameters.check_queue_spillover = True
        parameters.file_directory = './scenario_5'
        run(parameters=parameters)

    '''
    Scenario 6
    4 zones, demands from 3 zones well separated
    No route choice
    Plotting congestion in two zones + Congestion spillover
    '''
    def test_scenario_6(self):
        parameters = Parameters()
        parameters.check_queue_spillover = True
        parameters.congestion_links_jam_density = [100, 100, 100, 25]
        parameters.congestion_links_length = [100, 100, 100, 10]
        parameters.freeway_links_jam_density = [100, 100, 100, 25, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                                100, 100]
        parameters.freeway_links_length = [100, 100, 100, 10, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                           100]
        parameters.file_directory = './scenario_6'
        run(parameters=parameters)