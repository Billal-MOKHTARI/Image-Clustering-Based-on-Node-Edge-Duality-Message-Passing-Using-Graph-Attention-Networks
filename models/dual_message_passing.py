import torch.nn as nn
import torch
from message_passing import MessagePassing
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src import data_loader

class DualMessagePassing(nn.Module):

    def __init__(self, node_graph_order, 
                primal_in_features, 
                primal_out_features, 
                primal_depth,
                primal_index,
                dual_graph_order,
                dual_in_features,
                dual_out_features,
                dual_depth,
                dual_index, 
                layer_index,

                **kwargs):
        super(DualMessagePassing).__init__()
        self.node_message_passing_args = kwargs.get("node_message_passing_args", {})
        self.edge_message_passing_args = kwargs.get("edge_message_passing_args", {})

        self.node_graph_order = node_graph_order
        self.primal_in_features = primal_in_features
        self.primal_out_features = primal_out_features
        self.primal_depth = primal_depth
        self.primal_index = primal_index
        self.dual_graph_order = dual_graph_order
        self.dual_in_features = dual_in_features
        self.dual_out_features = dual_out_features
        self.dual_depth = dual_depth
        self.dual_index = dual_index
        self.layer_index = layer_index

        self.node_message_passing_block = MessagePassing(graph_order=self.node_graph_order, 
                                                   in_features=self.primal_in_features, 
                                                   out_features=self.primal_out_features, 
                                                   depth=self.primal_depth, 
                                                   layer_index=self.layer_index, 
                                                   **self.node_message_passing_args)
        
        self.edge_message_passing_block = MessagePassing(graph_order=self.dual_graph_order, 
                                                   in_features=self.dual_in_features, 
                                                   out_features=self.dual_out_features, 
                                                   depth=self.dual_depth, 
                                                   layer_index=self.layer_index, 
                                                   **self.edge_message_passing_args)

    def create_dual_adjacency_tensor(self, delimiter, mapper):
        tensor = torch.zeros(self.primal_out_features, len(self.dual_index), len(self.dual_index))

        for ind1 in self.dual_index:
            for ind2 in self.dual_index:
                ind1_split = ind1.split(delimiter)
                ind2_split = ind2.split(delimiter)

                intersection = data_loader.intersect_list(ind1_split, ind2_split)
                if len(intersection) == 1:
                    index_row, index_col, value = self.dual_index.index(ind1), self.dual_index.index(ind2), mapper[intersection[0]]
                    tensor[:, index_row, index_col] = value    

        return tensor 


        

    def create_primal_adjacency_tensor(self, primal_index, delimiter, mapper):

        pass

    def forward(self, node_x, edge_x, node_adjacency_tensor, edge_adjacency_tensor):
        node_output = self.node_message_passing_block(node_x, node_adjacency_tensor)
        edge_output = self.edge_message_passing_block(edge_x, edge_adjacency_tensor)


        pass