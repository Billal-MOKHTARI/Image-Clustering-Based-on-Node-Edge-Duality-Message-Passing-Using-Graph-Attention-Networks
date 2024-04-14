from torch import nn
import torch
import os
import sys
import numpy as np

import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class Linear2D(nn.Module):

    def __init__(self, depth, n_features):
        """
        Initializes a Linear2D object.

        Args:
            depth (int): The number of linear layers to create.
            n_features (int): The number of input and output features for each linear layer.

        Example:
            >>> model = Linear2D(depth=3, n_features=5)

            >>> mat1 = np.array([[1, 1, 0, 1, 1], 
                            [1, 1, 1, 0, 0], 
                            [0, 1, 1, 0, 1], 
                            [1, 0, 0, 1, 0],
                            [1, 0, 1, 0, 1]])

            >>> mat2 = np.array([[1, 0, 1, 1, 1], 
                            [0, 1, 1, 0, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 0, 1, 1, 0],
                            [1, 1, 1, 0, 1]])

            >>> mat3 = np.array([[1, 1, 0, 0, 1], 
                            [1, 1, 0, 1, 0], 
                            [0, 0, 1, 0, 1], 
                            [0, 1, 0, 1, 0],
                            [1, 0, 1, 0, 1]])

            >>> mat = torch.tensor(np.array([mat1, mat2, mat3], dtype=np.float32))
        """
        super(Linear2D, self).__init__()
        self.list_linear = []
        self.depth = depth

        for i in range(depth):
            self.list_linear.append(nn.Linear(n_features, n_features))
   

    def forward(self, x):
        assert len(x.shape) == 3, "The input tensor should be a 3D tensor"
        assert len(x) == self.depth, "The depth of the tensor should be equal to the depth of the model"



        list_product = []
        for channel, layer in zip(x, self.list_linear):
            list_product.append(layer(channel))
            
        return torch.stack(list_product)
    
