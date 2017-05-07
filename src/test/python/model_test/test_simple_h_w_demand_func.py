import unittest
from src.main.python.model_test.generate_demand import simple_h_w_demand_func

class test_simple_h_w_demand_func(unittest.TestCase):

    def test_run(self):
        func = simple_h_w_demand_func(100,200,10)
        self.assertTrue(func(50) == 0)
        self.assertTrue(func(100) == 0)
        self.assertTrue(func(125) == 250)
        self.assertTrue(func(150) == 500)
        self.assertTrue(func(175) == 250)
        self.assertTrue(func(200) == 0)
        self.assertTrue(func(250) == 0)