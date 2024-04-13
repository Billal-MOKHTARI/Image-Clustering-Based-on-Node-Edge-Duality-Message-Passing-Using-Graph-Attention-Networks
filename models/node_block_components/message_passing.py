from torch import nn
from torchsummary import summary
import torch
from linear_dot_2d import LinearDot2D
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import src.utils

class MessagePassing(nn.Module):
    def __init__(self, graph_order, in_features, out_features, layer_index, depth, **kwargs):
        super(MessagePassing, self).__init__()
        batch_norm_args_name = "batch_norm"
        activation_args_name = "activation"
        self.activation_exists = False

        self.layer_index = layer_index
        self.graph_order = graph_order
        self.depth = depth

        if batch_norm_args_name in kwargs.keys():
            batch_norm_args = kwargs[batch_norm_args_name]
        if activation_args_name in kwargs.keys():
            activation_args = kwargs[activation_args_name]
            layer = activation_args["layer"]
            activation_args = activation_args["args"]

            self.activation_exists = True
            self.activation = layer(**activation_args)

        self.linear_layer = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features, **batch_norm_args)
        self.linear_dot_2d = LinearDot2D(self.depth, n_features=graph_order)
        self.conv_1x1 = nn.Conv2d(in_channels=self.depth, out_channels=1, kernel_size=1)


    def get_layer_index(self):
        return self.layer_index

    def forward(self, x, adjacency_tensor):
        assert adjacency_tensor.shape[0] == self.depth, "The depth of the adjacency tensor should be equal to the depth of the LinearDot2D model"
        assert adjacency_tensor.shape[1] == adjacency_tensor.shape[2] == x.shape[0], "The adjacency tensor should be a 3D tensor with the same shape as the input tensor"
        # H(k) * W_h(k)
   
        x = self.linear_layer(x)
        a = self.linear_dot_2d(adjacency_tensor)
  

        h = list()
        for channel in a:
            h.append(torch.matmul(channel, x))

        h = torch.stack(h)
        h = self.conv_1x1(h)
        h = h.squeeze(0)
        # The batch norm layer is applied only when the batch size is greater than 1
        if h.shape[0] > 1:
            h = self.batch_norm(h)

        if self.activation_exists:
            h = self.activation(h)
        return h

# order = 5
# in_features = 4
# out_features = 7
# depth  = 3
mat1 = np.array([[1, 1, 0, 1, 1], 
                [1, 1, 1, 0, 0], 
                [0, 1, 1, 0, 1], 
                [1, 0, 0, 1, 0],
                [1, 0, 1, 0, 1]])

mat2 = np.array([[1, 0, 1, 1, 1], 
                [0, 1, 1, 0, 1], 
                [1, 1, 1, 1, 1], 
                [1, 0, 1, 1, 0],
                [1, 1, 1, 0, 1]])

mat3 = np.array([[1, 1, 0, 0, 1], 
                [1, 1, 0, 1, 0], 
                [0, 0, 1, 0, 1], 
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1]])

adjacency_tensor = torch.tensor(np.array([mat1, mat2, mat3]), dtype=torch.float32)
matrix = torch.tensor(np.array([[1, 2, 3, 4], [-1, 0, 3, 2], [4, 2, 2, 1], [0, 0.4, 0.1, 4], [0.4, 0.6, 2.5, 2]]), dtype=torch.float32)
   
model = MessagePassing(graph_order=5, in_features=4, out_features=3, layer_index=1, depth=3, batch_norm={"momentum": 0.1}, activation={"layer": nn.ReLU, "args": {}})
print(model(matrix, adjacency_tensor))
