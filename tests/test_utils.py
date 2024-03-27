import unittest
import sys
sys.path.append("../")
from src import utils

class TestRequiredKernel(unittest.TestCase):
    def test_equal_sizes_without_padding(self):
        size = 100
        self.assertEqual(utils.required_kernel(size, size, padding=0), 1)
    
    def test_equal_sizes_with_padding(self):
        size = 100
        self.assertEqual(utils.required_kernel(size, size, padding=1), 3)
        
    def test_smaller_output_size_without_padding(self):
        self.assertEqual(utils.required_kernel(100, 50, padding=0), 51)
    
    def test_smaller_output_size_with_padding(self):
        self.assertEqual(utils.required_kernel(100, 50, padding=2), 55)
        
    def test_smaller_output_size_with_padding_and_stride(self):
        self.assertEqual(utils.required_kernel(100, 50, padding=1, stride=2), 4)
    
        
        
if __name__ == "__main__":
    unittest.main()