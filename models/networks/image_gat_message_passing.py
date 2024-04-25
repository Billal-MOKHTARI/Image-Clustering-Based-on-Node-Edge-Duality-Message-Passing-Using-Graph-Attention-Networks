import torch
from torch import nn
from message_passing import MessagePassing
import numpy as np
import constants
from typing import List
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import training.trainer

from src import utils
from training import trainer
from project_consts import NEPTUNE_MANAGER, DATA_VISUALIZATION_RUN
class ImageGATMessagePassing(nn.Module):
    def __init__(self, graph_order, depth, layer_sizes, loss: nn.Module, loss_coeffs: List[float], **kwargs):
        super(ImageGATMessagePassing, self).__init__()
        assert all(0 <= loss_coeff <= 1 for loss_coeff in loss_coeffs), "Loss coefficients must be between 0 and 1"
        self.encoder_args = kwargs.get("encoder_args", {})
        self.decoder_args = kwargs.get("decoder_args", {})
        self.graph_order = graph_order
        self.depth = depth
        self.layer_sizes = layer_sizes
        self.encoder_layers = self.encoder()
        self.decoder_layers = self.decoder()
        self.loss = loss
        self.loss_coeffs = loss_coeffs
 
    def encoder(self):
        encoder_layers = nn.ModuleDict()
        for i in range(len(self.layer_sizes)-1):
            encoder_layers[f"enc_{i}"] = MessagePassing(graph_order=self.graph_order, 
                                                         in_features=self.layer_sizes[i], 
                                                         out_features=self.layer_sizes[i+1], 
                                                         depth=self.depth, 
                                                         layer_index=f"enc_{i}",
                                                         **self.encoder_args)

        return encoder_layers
    
    def decoder(self):
        decoder_layers = nn.ModuleDict()
        length = len(self.layer_sizes)-1
        for i in range(length):
            decoder_layers[f"dec_{i}"] = MessagePassing(graph_order=self.graph_order, 
                                                         in_features=self.layer_sizes[length-i], 
                                                         out_features=self.layer_sizes[length-i-1], 
                                                         depth=self.depth, 
                                                         layer_index=f"dec_{i}",
                                                         **self.decoder_args)
        
        return decoder_layers

    def forward(self, x, adjacency_tensor):
        enc_outputs = []
        dec_outputs = []

        for name, layer in self.encoder_layers.items():
            enc_outputs.append(x)
            x = layer(x, adjacency_tensor)
        
        for name, layer in self.decoder_layers.items():
            x = layer(x, adjacency_tensor)
            dec_outputs.append(x)
 
        # Inverse the decoder outputs to get the values in order with the encoder outputs
        dec_outputs = dec_outputs[::-1]

        loss = torch.mean(torch.stack([loss_coeff*self.loss(enc, dec) for enc, dec, loss_coeff in zip(enc_outputs, dec_outputs, self.loss_coeffs)]))
        return x, loss

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

mat = torch.tensor(np.array([mat1, mat2, mat3]), dtype=constants.FLOATING_POINT)
x = torch.randn(5, 10, dtype=constants.FLOATING_POINT)


model = ImageGATMessagePassing(graph_order=5, 
                               depth=3, 
                               layer_sizes=[10, 8, 6, 4], 
                               loss=nn.MSELoss(), 
                               loss_coeffs=[1, 1, 1])


trainer.image_gat_mp_trainer(model, 
                             x,
                            1000, 
                            torch.optim.Adam, 
                            mat)