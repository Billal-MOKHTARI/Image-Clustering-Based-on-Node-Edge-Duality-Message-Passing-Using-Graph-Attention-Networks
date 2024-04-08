from torch import nn
import torch
from mp_node_encoder import MPNodeEncoder
from node_adjacency_matrix_aggregator import NodeAdjacencyMatrixAgregator
class NodeEncoder(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                layer_index, 
                num_embeddings, 
                embedding_dim, 
                matrix_dim, 
                **kwargs):
        
        super(NodeEncoder, self).__init__()

        self.mp_node_encoder_args = kwargs.get("mp_node_encoder_args", {})
        self.node_activation = kwargs.get("node_activation", None)

        if self.node_activation is not None :
            self.activation = self.node_activation["activation"]
            self.activation_args = self.node_activation["activation_args"]

            self.node_activation_layer = self.activation(self.activation_args)

        self.mp_node_encoder = MPNodeEncoder(in_features, out_features, layer_index, **self.mp_node_encoder_args)
        self.node_adjacency_matrix_aggregator = NodeAdjacencyMatrixAgregator(num_embeddings, embedding_dim, matrix_dim)

    def forward(self, adj, measure_matrix, node_features):
        assert adj.shape[0] == self.matrix_dim, f"The batch size should be the same as the dimension of adjacency matrix, which represents the number of the nodes."
        adj_mat = self.node_adjacency_matrix_aggregator(adj, measure_matrix)
        h = self.mp_node_encoder(node_features)

        x = torch.matmul(adj_mat, h)
        
        if self.node_activation is not None:
            x = self.node_activation_layer(x)

        return x

        
        
