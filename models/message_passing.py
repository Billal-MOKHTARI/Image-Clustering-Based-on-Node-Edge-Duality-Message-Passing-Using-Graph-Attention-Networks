from torch import nn
from torchsummary import summary
import torch
from custom_layers.linear_2d import Linear2D
import numpy as np
import os
import sys
import constants

class MessagePassing(nn.Module):
    """
    MessagePassing class represents a message passing layer in a graph attention network.
    It takes an input tensor and an adjacency tensor as input and applies linear transformations,
    graph convolutions, batch normalization, and activation functions to produce the output tensor.

    Args:
        graph_order (int): The order of the graph.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        layer_index (int): The index of the layer.
        depth (int): The depth of the adjacency tensor.
        **kwargs: Additional keyword arguments for batch normalization and activation.

    Attributes:
        activation_exists (bool): Indicates whether an activation function is specified.
        layer_index (int): The index of the layer.
        graph_order (int): The order of the graph.
        depth (int): The depth of the adjacency tensor.
        activation (nn.Module): The activation function.
        linear_layer (nn.Linear): The linear transformation layer.
        batch_norm (nn.BatchNorm1d): The batch normalization layer.
        linear_dot_2d (Linear2D): The 2D linear transformation layer.
        conv_1x1 (nn.Conv2d): The 1x1 convolutional layer.

    Methods:
        get_layer_index(): Returns the layer index.
        forward(x, adjacency_tensor): Performs forward pass through the message passing layer.

    Example:
        >>> # order = 5
        >>> # in_features = 4
        >>> # out_features = 7
        >>> # depth  = 3
        >>> mat1 = np.array([[1, 1, 0, 1, 1], 
        >>>                 [1, 1, 1, 0, 0], 
        >>>                 [0, 1, 1, 0, 1], 
        >>>                 [1, 0, 0, 1, 0],
        >>>                 [1, 0, 1, 0, 1]])

        >>> mat2 = np.array([[1, 0, 1, 1, 1], 
        >>>                 [0, 1, 1, 0, 1], 
        >>>                 [1, 1, 1, 1, 1], 
        >>>                 [1, 0, 1, 1, 0],
        >>>                 [1, 1, 1, 0, 1]])

        >>> mat3 = np.array([[1, 1, 0, 0, 1], 
        >>>                 [1, 1, 0, 1, 0], 
        >>>                 [0, 0, 1, 0, 1], 
        >>>                 [0, 1, 0, 1, 0],
        >>>                 [1, 0, 1, 0, 1]])

        >>> adjacency_tensor = torch.tensor(np.array([mat1, mat2, mat3]), dtype=torch.float32)
        >>> matrix = torch.tensor(np.array([[1, 2, 3, 4], [-1, 0, 3, 2], [4, 2, 2, 1], [0, 0.4, 0.1, 4], [0.4, 0.6, 2.5, 2]]), dtype=torch.float32)
        
        >>> model = MessagePassing(graph_order=5, in_features=4, out_features=3, layer_index=1, depth=3, batch_norm={"momentum": 0.1}, activation={"layer": nn.ReLU, "args": {}})
        >>> print(model(matrix, adjacency_tensor))

        >>> # Create a MessagePassing instance
        >>> model = MessagePassing(graph_order=5, in_features=4, out_features=3, layer_index=1, depth=3, batch_norm={"momentum": 0.1}, activation={"layer": nn.ReLU, "args": {}})
        
        >>> # Perform forward pass
        >>> output = model(input_tensor, adjacency_tensor) 
    """

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
        self.linear_dot_2d = Linear2D(self.depth, n_features=graph_order)
        self.conv_1x1 = nn.Conv2d(in_channels=self.depth, out_channels=1, kernel_size=1)

        # Copy the model to specified device
        self.to(constants.DEVICE)

    def get_layer_index(self):
        """
        Returns the layer index.

        Returns:
            int: The layer index.
        """
        return self.layer_index

    def set_layer_index(self, layer_index):
        """
        Sets the layer index.

        Args:
            layer_index (int): The new layer index.
        """
        self.layer_index = layer_index

    def forward(self, x, adjacency_tensor):
        """
        Performs forward pass through the message passing layer.

        Args:
            x (torch.Tensor): The input tensor.
            adjacency_tensor (torch.Tensor): The adjacency tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        assert adjacency_tensor.shape[0] == self.depth, "The depth of the adjacency tensor should be equal to the depth of the LinearDot2D model"
        assert adjacency_tensor.shape[1] == adjacency_tensor.shape[2] == x.shape[0], "The adjacency tensor should be a 3D tensor with the same shape as the input tensor"
        
        # Copy tensors to specified device
        x.to(constants.DEVICE)
        adjacency_tensor.to(constants.DEVICE)

        # Convert tensors to floating point
        x.to(constants.FLOATING_POINT)
        adjacency_tensor.to(constants.FLOATING_POINT)

        # Perform linear transformation
        x = self.linear_layer(x)
        a = self.linear_dot_2d(adjacency_tensor)

        # Perform graph convolution
        h = list()
        for channel in a:
            h.append(torch.matmul(channel, x))

        # Perform 1x1 convolution
        h = torch.stack(h)
        h = self.conv_1x1(h)
        h = h.squeeze(0)

        # Perform batch normalization
        if h.shape[0] > 1:
            h = self.batch_norm(h)

        # Perform activation
        if self.activation_exists:
            h = self.activation(h)
        return h

