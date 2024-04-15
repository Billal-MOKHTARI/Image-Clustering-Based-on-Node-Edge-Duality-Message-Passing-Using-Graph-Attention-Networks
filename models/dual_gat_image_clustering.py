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



##############################################################

def create_dual_adjacency_tensor(primal_adjacency_tensor, primal_index, delimiter):
        
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
        def create_dual(matrix, delimiter):
        
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
                        dual_matrix.loc[ind1, ind2] = 1
                        
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

            return sorted(new_index)

        dual_index = create_dual_index(primal_adjacency_tensor, primal_index, delimiter)
        channels = utils.extract_channels(primal_adjacency_tensor, primal_index, primal_index)

        dual_channels = []

        ind = 0
        for channel in channels:
            
            dual_channel = create_dual(channel, delimiter)
            dual_channel = sort_and_fill_matrix(dual_channel, dual_index)
            dual_channels.append(dual_channel)
            ind += 1
        
        dual_adjacency_tensor = torch.tensor(np.array(dual_channels), dtype=constants.FLOATING_POINT)
        dual_nodes = []
        for ind in dual_index:
            row, col = ind.split(delimiter)
            row, col = primal_index.index(row), primal_index.index(col)
            dual_nodes.append(primal_adjacency_tensor[:, row, col])
        
        dual_nodes = torch.stack(dual_nodes, dim=0)
        dual_nodes.to(dtype=constants.FLOATING_POINT)
        
        return dual_index, dual_adjacency_tensor, dual_nodes


##############################################################


class DualGATImageClustering(nn.Module):
    def __init__(self, 
                primal_index,
                dual_index,
                n_objects,
                backbone = 'vgg16', 
                in_image_size = (3, 224, 224), 
                margin_expansion_factor=6,
                primal_mp_layer_inputs=[728, 600, 512, 400, 256],  
                dual_mp_layer_inputs=[1000, 728, 600, 512, 400],
                delimiter="_",
                **kwargs):
        
        super(DualGATImageClustering, self).__init__()
        
        # Extract additional keyword arguments
        image_encoder_args = kwargs.get("image_encoder_args", {})
        dual_message_passing_args = kwargs.get("dual_message_passing_args", {})
        
        # Store input size and primal index
        self.image_size = in_image_size
        self.primal_index = primal_index 
        self.dual_index = dual_index
        self.delimiter = delimiter 
        
        self.image_encoder = ImageEncoder(input_shape=in_image_size, 
                                          model=backbone, 
                                          margin_expansion_factor=margin_expansion_factor, 
                                          **image_encoder_args)
        
        self.out_img_size = utils.get_output_model(self.image_encoder, in_image_size)
        
        # Create primal layer inputs
        self.primal_mp_layer_inputs = []
        self.primal_mp_layer_inputs.append(self.out_img_size)
        self.primal_mp_layer_inputs.extend(primal_mp_layer_inputs)

        # Create dual layer inputs
        self.dual_mp_layer_inputs = []
        self.dual_mp_layer_inputs.append(n_objects)
        self.dual_mp_layer_inputs.extend(dual_mp_layer_inputs)
              
        # Create dual message passing depths
        self.dual_depths = []
        self.dual_depths.append(n_objects)
        self.dual_depths.extend(primal_mp_layer_inputs)
        
        # Create dual message passing layers
        self.dmp_layers = []
        for i in range(len(self.primal_mp_layer_inputs)-1):
            self.dmp_layers.append(DualMessagePassing(primal_in_features=self.primal_mp_layer_inputs[i], 
                                                      primal_out_features=self.primal_mp_layer_inputs[i+1], 
                                                      primal_index=self.primal_index,
                                                      primal_depth=self.dual_mp_layer_inputs[i],
                                                    
                                                      dual_in_features=self.dual_mp_layer_inputs[i],
                                                      dual_out_features=self.dual_mp_layer_inputs[i+1],
                                                      dual_index=self.dual_index,
                                                      dual_depth=self.dual_depths[i],
                                                      layer_index=i,
                                                      delimiter=self.delimiter,
                                                      **dual_message_passing_args))

    def forward(self, imgs, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        # Encode images to embeddings
        primal_nodes = self.image_encoder(imgs)


        for layer in self.dmp_layers:
            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)
            
            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]
            print(f"Primal Adjacency Tensor : {primal_adjacency_tensor.shape}")
            print(f"Dual Adjacency Tensor : {dual_adjacency_tensor.shape}")
            
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor
            
num_imgs = 5
n_objects = 4
img = torch.randn(num_imgs, 3, 224, 224)
primal_index = ["1", "2", "3", "4", "5"]

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

mat4 = np.array([[1, 1, 0, 0, 1], 
                [1, 1, 0, 1, 0], 
                [0, 0, 1, 0, 1], 
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1]])

primal_adjacency_tensor = torch.tensor(np.array([mat1, mat2, mat3, mat4]), dtype=constants.FLOATING_POINT)
dual_index, dual_adjacency_tensor, dual_nodes = create_dual_adjacency_tensor(primal_adjacency_tensor, primal_index, "_")
print(dual_index)
model = DualGATImageClustering(primal_index=primal_index, dual_index=dual_index, n_objects=n_objects)
model(img, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
