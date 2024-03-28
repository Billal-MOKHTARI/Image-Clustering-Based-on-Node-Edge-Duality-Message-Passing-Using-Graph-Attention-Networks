import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import unittest
from src import py_model_manager
from torchvision.models import vgg16
from torch.nn import AdaptiveAvgPool2d

class TestPyModelManager(unittest.TestCase):
    def setUp(self):
        self.vgg = py_model_manager.models.vgg16(pretrained=True)
        self.py_model_manager = py_model_manager.PyModelManager(self.vgg)
        
    def test_get_named_layers(self):
        for key, value in self.py_model_manager.get_named_layers().items():
            self.assertEqual(value, eval("self.vgg."+key))
    
    def test_get_list_layers(self):
        for l1, l2 in zip(self.py_model_manager.get_list_layers(), self.vgg.children()):
            self.assertEqual(l1, l2)
        
    def test_get_layer_by_name(self):
        for layer_name in self.py_model_manager.get_named_layers().keys():
            self.assertEqual(self.py_model_manager.get_layer_by_name(layer_name), eval("self.vgg."+layer_name))
            if layer_name != 'avgpool':
                for i in range(len(eval("self.vgg."+layer_name))):
                    self.assertEqual(self.py_model_manager.get_layer_by_name(layer_name, i), eval("self.vgg."+layer_name)[i])  

    def test_get_layer_by_index(self):
        for i, layer in zip(range(len(self.py_model_manager.get_list_layers())), self.vgg.children()):
            self.assertEqual(self.py_model_manager.get_layer_by_index(i), layer)
            
            if not isinstance(layer, AdaptiveAvgPool2d):
                for j in range(len(layer)):
                    self.assertEqual(self.py_model_manager.get_layer_by_index(i, j), layer[j])
    
    def test_delete_layer_by_name_without_index(self):
        for layer_name in self.py_model_manager.get_named_layers().keys():
            self.py_model_manager.delete_layer_by_name(layer_name)
            self.assertFalse(hasattr(self.vgg, layer_name))
    
    # def test_delete_layer_by_name_with_index(self):
    #     for layer_name in self.py_model_manager.get_named_layers().keys():
    #         if not isinstance(layer_name, AdaptiveAvgPool2d):
    #             for j in range(len(eval("self.vgg."+layer_name))):
    #                 self.py_model_manager.delete_layer_by_name(layer_name, j)
    #                 print(self.)
    #                 self.assertNotIn(eval("self.vgg."+layer_name)[j], eval("self.vgg."+layer_name))
    
if __name__ == "__main__":
    unittest.main()