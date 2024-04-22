import torch.nn as nn
import torch
from .message_passing import MessagePassing
import pandas as pd
import numpy as np
from . import constants
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import utils

class DualMessagePassing(nn.Module):
    """
        dual_graph_order (int): Order of the dual graph.
        dual_in_features (int): Number of input features for the dual graph.
        dual_out_features (int): Number of output features for the dual graph.
        dual_index (list): List of indices for the dual graph.
        layer_index (int): Index of the layer.
        delimiter (str): Delimiter used to split indices.
        node_message_passing_block (MessagePassing): Node message passing block.
        edge_message_passing_block (MessagePassing): Edge message passing block.

    Methods:
        create_dual_adjacency_tensor(mapper): Creates the dual adjacency tensor.
        create_primal_adjacency_tensor(mapper): Creates the primal adjacency tensor.
        forward(node_x, edge_x, primal_adjacency_tensor, dual_adjacency_tensor): Performs forward pass.

    Returns:
        dict: Dictionary containing the primal and dual outputs and adjacency tensors. primal and dual outputs and adjacency tensors.
    
    Example :
    >>>     model = DualMessagePassing(
    >>>                             primal_in_features=3, 
    >>>                             primal_out_features=7, 
    >>>                             depth=3,
    >>>                             primal_index=["n1", "n2", "n3", "n4"],
    >>>                             dual_in_features=5,
    >>>                             dual_out_features=3,
    >>> 
    >>>                             dual_index=["n1_n2", "n1_n3", "n2_n4"], 
    >>>                             layer_index=1,
    >>>                             delimiter="_",
    >>>                             node_message_passing_args={"batch_norm": {"momentum": 0.1}, "activation": {"layer": nn.ReLU, "args": {}}},
    >>>                             edge_message_passing_args={"batch_norm": {"momentum": 0.1}, "activation": {"layer": nn.ReLU, "args": {}}}
    >>>                             )

    >>>     node_x = torch.tensor(np.array([[1, 2, 3], [4, 6, 5], [7, 8, 9], [10, 11, 12]]), dtype=torch.float32)
    >>>     edge_x = torch.tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]), dtype=torch.float32)
    >>>     primal_adjacency_tensor = torch.tensor(np.array([[[1, 1, 1, 0], 
    >>>                                                     [1, 1, 0, 0], 
    >>>                                                     [1, 0, 1, 0], 
    >>>                                                     [0, 0, 0, 1]], 
    >>>                                                     [[1, 1, 1, 0], 
    >>>                                                     [1, 1, 0, 1], 
    >>>                                                     [1, 0, 1, 0], 
    >>>                                                     [0, 1, 0, 1]], 
    >>>                                                     [[1, 1, 0, 0], 
    >>>                                                     [1, 1, 0, 1], 
    >>>                                                     [0, 0, 1, 0], 
    >>>                                                     [0, 1, 0, 1]]]), dtype=torch.float32)
    >>>     dual_adjacency_tensor = torch.tensor(np.array([[[1, 2, 3], 
    >>>                                                     [2, 1, 1], 
    >>>                                                     [3, 1, 0]], 
    >>>                                                 [[1, 0, 1], 
    >>>                                                     [0, 1, 0], 
    >>>                                                     [1, 0, 1]], 
    >>>                                                 [[1, 0, 1], 
    >>>                                                     [0, 1, 1], 
    >>>                                                     [1, 1, 1]]]), dtype=torch.float32)
    >>>     result = model(node_x, edge_x, primal_adjacency_tensor, dual_adjacency_tensor)
    >>>     print(result)

    """

    def __init__(self,
                primal_in_features, 
                primal_out_features, 
                primal_index,
                primal_depth,
                dual_depth,
                dual_in_features,
                dual_out_features,
                dual_index, 
                layer_index,
                delimiter="_",
                **kwargs):
        
        super(DualMessagePassing, self).__init__()
        self.node_message_passing_args = kwargs.get("node_message_passing_args", {})
        self.edge_message_passing_args = kwargs.get("edge_message_passing_args", {})

        self.node_graph_order = len(primal_index)
        self.primal_in_features = primal_in_features
        self.primal_out_features = primal_out_features
        self.primal_index = primal_index
        self.primal_depth = primal_depth

        self.dual_graph_order = len(dual_index)
        self.dual_in_features = dual_in_features
        self.dual_out_features = dual_out_features
        self.dual_index = dual_index
        self.dual_depth = dual_depth
 
        self.layer_index = layer_index
        self.delimiter = delimiter

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

        self.to(constants.DEVICE)

    def create_dual_adjacency_tensor(self, mapper):
        """
        Creates the dual adjacency tensor.

        Args:
            mapper (pd.DataFrame): DataFrame containing the mapping between indices and values.

        Returns:
            torch.Tensor: Dual adjacency tensor.
        """
        tensor = torch.zeros(mapper.shape[1], len(self.dual_index), len(self.dual_index))

        for ind1 in self.dual_index:
            for ind2 in self.dual_index:
                ind1_split = ind1.split(self.delimiter)
                ind2_split = ind2.split(self.delimiter)

                intersection = utils.intersect_list(ind1_split, ind2_split)
                if len(intersection) == 1:
                    index_row, index_col, value = self.dual_index.index(ind1), self.dual_index.index(ind2), mapper.loc[intersection[0]]
                    tensor[:, index_row, index_col] = torch.tensor(value)
                elif len(intersection) == 2:
                    tensor[:, self.dual_index.index(ind1), self.dual_index.index(ind2)] = torch.tensor(np.ones(mapper.shape[1]))

        return tensor

    def create_primal_adjacency_tensor(self, mapper):  
        """
        Creates the primal adjacency tensor.

        Args:
            mapper (pd.DataFrame): DataFrame containing the mapping between indices and values.

        Returns:
            torch.Tensor: Primal adjacency tensor.
        """
        tensor = torch.zeros(mapper.shape[1], len(self.primal_index), len(self.primal_index))
        
        mapper_index = list(mapper.index)
        for ind in mapper_index:
            ind_split = ind.split(self.delimiter)
            ind_row, ind_col = self.primal_index.index(ind_split[0]), self.primal_index.index(ind_split[1])

            tensor[:, ind_row, ind_col] = torch.tensor(mapper.loc[ind])
            tensor[:, ind_col, ind_row] = torch.tensor(mapper.loc[ind])
        
        for i in range(len(self.primal_index)):
            tensor[:, i, i] = torch.tensor(np.ones(mapper.shape[1]))

        return tensor

    def forward(self, node_x, edge_x, primal_adjacency_tensor, dual_adjacency_tensor):
        """
        Performs forward pass.

        Args:
            node_x (torch.Tensor): Input tensor for the primal graph.
            edge_x (torch.Tensor): Input tensor for the dual graph.
            primal_adjacency_tensor (torch.Tensor): Primal adjacency tensor.
            dual_adjacency_tensor (torch.Tensor): Dual adjacency tensor.

        Returns:
            dict: Dictionary containing the primal and dual outputs and adjacency tensors.
        """
        
        primal_output = self.node_message_passing_block(node_x, primal_adjacency_tensor)
        dual_output = self.edge_message_passing_block(edge_x, dual_adjacency_tensor)
        
        primal_adjacency_tensor = self.create_primal_adjacency_tensor(pd.DataFrame(dual_output.detach().numpy(), index=self.dual_index))
        dual_adjacency_tensor = self.create_dual_adjacency_tensor(pd.DataFrame(primal_output.detach().numpy(), index=self.primal_index))


        result = {
                    "primal": 
                            {"nodes": primal_output, 
                             "adjacency_tensor": primal_adjacency_tensor
                            },
                    "dual": 
                            {
                            "nodes": dual_output, 
                             "adjacency_tensor": dual_adjacency_tensor
                            }
                }

        return result
