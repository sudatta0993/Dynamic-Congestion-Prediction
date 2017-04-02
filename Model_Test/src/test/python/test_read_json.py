import unittest
import os
import json

class test_read_json(unittest.TestCase):

    def test_generate_json(self):
        dict = {'min_intervals':5,
                'num_bins':288,
                'num_zones':4,
                'demand_start_times':[0, 480, 960],
                'demand_end_times':[240, 720, 1200],
                'demand_slopes':[0.1,0.1,0.1],
                'congestion_links_capacity':[10,10,10,5],
                'threshold_output_for_congestion':[1,1,1,1],
                'congestion_links_fftt':[20,20,20,20],
                'congestion_links_jam_density':[100,100,100,100],
                'congestion_links_length':[100,100,100,100],
                'freeway_links_capacity':[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,20, 20, 20, 20],
                'freeway_links_fftt':[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
                'freeway_links_jam_density':[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
                'freeway_links_length':[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
                'congestion_nn_smoothening_number':[10,10,10,10],
                'check_route_choice':False,
                'plot_congestion_io_curves':True,
                'plot_demand_congestion_curves':True,
                'plot_route_choice_io_curves':False,
                'check_queue_spillover':False,
                'file_directory':'./scenario_1',
                'get_curves_data':False}
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(test_dir+'/test_input.json', 'w') as fp:
            json.dump(dict,fp)
        fp.close()
        self.assertTrue(os.path.isfile(test_dir+'/test_input.json'))

    def test_read_json(self):
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(test_dir + '/test_input.json') as json_file:
            data = json.load(json_file)
            self.assertTrue(data['min_intervals'] == 5)
            self.assertTrue(data['freeway_links_fftt'] == [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])
            self.assertTrue(data['check_route_choice'] == False)
