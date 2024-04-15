import torch
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
dual_index, dual_adjacency_tensor, dual_nodes = utils.create_dual_adjacency_tensor(primal_adjacency_tensor, primal_index, "_")

model = DualGATImageClustering(primal_index=primal_index, dual_index=dual_index, n_objects=n_objects)
model(img, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
