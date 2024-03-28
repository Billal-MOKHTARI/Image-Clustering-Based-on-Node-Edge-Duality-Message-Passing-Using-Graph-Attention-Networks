import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import unittest
from src import py_model_manager

class TestPyModelManager(unittest.TestCase):
    def setUp(self):
        self.vgg = py_model_manager.models.vgg16(pretrained=True)
        self.py_model_manager = py_model_manager.PyModelManager(self.vgg)
        
    def test_get_named_layers(self):
        for key, value in self.py_model_manager.get_named_layers().items():
            self.assertEqual(value, eval("self.vgg."+key))
        
    def test_get_layer_by_name(self):
        for layer_name in self.py_model_manager.get_named_layers().keys():
            self.assertEqual(self.py_model_manager.get_layer_by_name(layer_name), eval("self.vgg."+layer_name))
                            

    
if __name__ == "__main__":
    unittest.main()