from torch import nn
import constants
from dual_message_passing import DualMessagePassing
from image_encoder import ImageEncoder
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src import utils

class DualGATImageClustering(nn.Module):
    def __init__(self, 
                image_size, 
                backbone, 
                index,
                margin_expansion_factor=6,
                mp_layer_inputs=[728, 600, 512, 400, 256],  
                delimiters="_",
                **kwargs):
        super(DualGATImageClustering, self).__init__()
        image_encoder_args = kwargs.get("image_encoder_args", {})
        
        self.image_size = image_size
        self.image_encoder = ImageEncoder(input_shape=image_size, 
                                          model=backbone, 
                                          margin_expansion_factor=margin_expansion_factor, 
                                          **image_encoder_args)
        self.index = index
        
    def create_dual_adjacency_tensor(primal_adjaceny_tensor, primal_index, mapper):
        def create_index(matrix, delimiter):
            assert matrix.shape[0] == matrix.shape[1], "The matrix should be square"
            rank = matrix.shape[0]
            columns = matrix.columns
            new_index = []

            assert [delimiter not in column for column in columns], "The delimiter should not be present in any column"

            np_matrix = matrix.to_numpy()

            is_symmetric = np.allclose(np_matrix, np_matrix.T)
            if not is_symmetric:
                k = 0

            for i in range(rank):
                if is_symmetric:
                    k = i
                for j in range(k+1, rank):
                    if np_matrix[i][j] != 0:
                        new_index.append(f"{columns[i]}{delimiter}{columns[j]}")

            return new_index
        
        def create_dual_index(primal_adjacency_tensor, primal_index, delimiter):
            assert len(primal_adjacency_tensor.shape) == 3, "The tensor should be a 3D tensor"
            assert primal_adjacency_tensor.shape[1] == primal_adjacency_tensor.shape[2], "The tensor should be square"

            channels = utils.extract_channels(primal_adjacency_tensor.detach().numpy(), primal_index, primal_index)

            new_index = list()
            for channel in channels:
                new_index = list(set(new_index).union(set(create_index(channel, delimiter))))
            
            return new_index
        
    def forward(self, imgs):
        imgs = self.image_encoder(imgs)
        