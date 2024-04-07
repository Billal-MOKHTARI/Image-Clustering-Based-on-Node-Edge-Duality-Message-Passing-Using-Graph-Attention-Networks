from torch import nn
from torchsummaryX import summary
import torch
class NodeAdjacencyMatrixAgregator(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, matrix_dim, **kwargs):
        super(NodeAdjacencyMatrixAgregator, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.matrix_dim = matrix_dim

    def forward(self, x, measure_matrix):
        # The measure matrix might me degree matrix, closeness matrix or any other matrix
        assert x.shape[1] == self.matrix_dim == measure_matrix.shape[0] == measure_matrix.shape[1], f"The measure matrix should be square of dimension {self.matrix_dim}"
        assert measure_matrix.shape[0] == self.matrix_dim == measure_matrix.shape[1], f"The measure matrix should be square of dimension {self.matrix_dim}"
        x = self.embedding_layer(x)
        x = x.squeeze()
        
        sqrt_measure_matrix = torch.sqrt(measure_matrix)
        inv_sqrt_measure_matrix = torch.inverse(sqrt_measure_matrix)

        print(inv_sqrt_measure_matrix)
        x = torch.matmul(inv_sqrt_measure_matrix, x)
        x = torch.matmul(x, inv_sqrt_measure_matrix)

        return x

# mat = torch.tensor([[0, 2, 1, 5], [2, 2, 1, 3], [0, 1, 2, 4], [1, 2, 3, 4]])
# measure_matrix = torch.tensor([[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 4]])
# model = NodeAdjacencyMatrixAgregator(6, 1, 4)
# print(model(mat, measure_matrix))

