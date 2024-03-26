import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeMessagePassingBlock(nn.Module):
    def __init__(self, input_shape, num_layers):
        super(NodeMessagePassingBlock, self).__init__()

        self.input_shape = input_shape
        self.num_layers = num_layers

        
        
        
    def forward(self, x, adjacency_matrix):

        return