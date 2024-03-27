import unittest
import sys
import os
# Add the parent directory of the current file to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from models.node_block_components import node_transformer
import models.constants




class TestNodeTransformer(unittest.TestCase):
    def setUp(self):
        self.input_shape = (0, 224, 224)
        self.model = "resnet18"
        self.margin_expansion_factor = 6
        self.kwargs = {"pretrained": True, "stride": (1, 1), "padding": (0, 0)}
        self.node_transformer = node_transformer.NodeTransformer(self.input_shape, self.model, self.margin_expansion_factor, **self.kwargs)
        print(self.node_transformer)
        
if __name__ == '__main__':
    unittest.main()