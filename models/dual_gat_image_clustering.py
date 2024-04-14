import torch
from torch import nn
import constants
from dual_message_passing import DualMessagePassing
from image_encoder import ImageEncoder
import numpy as np
import os
import sys
import pandas as pd
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
        
    def create_dual_adjacency_tensor(self, primal_adjacency_tensor, primal_index, mapper):
        
        def sort_and_fill_matrix(matrix, large_index):
            # Sort the index of the matrix according to the large index
            sorted_index = sorted(matrix.index, key=lambda x: large_index.index(x))
            sorted_columns = sorted(matrix.columns, key=lambda x: large_index.index(x))
            
            # Reindex the matrix
            matrix = matrix.reindex(index=sorted_index, columns=sorted_columns)
            
            # Add empty rows and columns where indexes of the matrix are missing
            missing_rows = [index for index in large_index if index not in matrix.index]
            missing_columns = [index for index in large_index if index not in matrix.columns]
            
            for row in missing_rows:
                matrix.loc[row] = [0] * len(matrix.columns)
                
            for col in missing_columns:
                matrix[col] = [0] * len(matrix.index)
            
            # Reorder rows and columns to match large_index
            matrix = matrix.reindex(index=large_index, columns=large_index)
            
            return matrix
        def create_dual(matrix, delimiter, mapper):
            # assert all(isinstance(i, (float, int)) for i in mapper.values()), "All elements must be floats"
            dual_index = create_index(matrix, delimiter)
            size = len(dual_index)

            init_data = np.zeros((size, size))
            dual_matrix = pd.DataFrame(init_data, index=dual_index, columns=dual_index)

            for ind1 in dual_index:
                for ind2 in dual_index:
                    ind1_split = ind1.split(delimiter)
                    ind2_split = ind2.split(delimiter)

                    intersection = utils.intersect_list(ind1_split, ind2_split)

                    if len(intersection) == 2:
                        dual_matrix.loc[ind1, ind2] = 1
                    
                    elif intersection != []:
                        dual_matrix.loc[ind1, ind2] = mapper.loc[intersection[0]]
                        
            return dual_matrix
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

        assert mapper.shape[0] == len(primal_index), "The number of rows in the mapper should be equal to the number of indexes"
        
        dual_index = create_dual_index(primal_adjacency_tensor, primal_index, self.delimiter)

        channels = utils.extract_channels(primal_adjacency_tensor, primal_index, primal_index)
        assert mapper.shape[1] == len(channels), "The number of columns in the mapper should be equal to the number of channels"
        dual_channels = []

        ind = 0
        for channel in channels:
            channel_mapper = mapper.iloc[:, ind]
            dual_channel = create_dual(channel, self.delimiter, channel_mapper)
            dual_channel = sort_and_fill_matrix(dual_channel, dual_index)

            dual_channels.append(dual_channel)
            ind += 1
        
        return dual_index, torch.tensor(np.array(dual_channels))

    def forward(self, imgs):
        imgs = self.image_encoder(imgs)
        