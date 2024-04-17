import torch
from torch import nn

from dual_message_passing import DualMessagePassing
from image_encoder import ImageEncoder
import os
import sys
from src import utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from torch import nn
import constants
import numpy as np

class DualGATImageClustering(nn.Module):
    def __init__(self, 
                primal_index,
                dual_index,
                n_objects,
                criterion = nn.MSELoss(),
                primal_criterion_weights = [1, 1, 1, 1, 1],
                dual_criterion_weights = [1, 1, 1, 1, 1],
                backbone='vgg16', 
                in_image_size=(4, 224, 224), 
                margin_expansion_factor=6,
                primal_mp_layer_inputs=[728, 600, 512, 400, 256],  
                dual_mp_layer_inputs=[1000, 728, 600, 512, 400],
                delimiter="_",
                **kwargs):
        """
        DualGATImageClustering model for image clustering based on node-edge duality message passing using Graph Attention Networks.
        
        Args:
            primal_index (list): List of primal node indices.
            dual_index (list): List of dual node indices.
            n_objects (int): Number of objects in the graph.
            backbone (str): Backbone model architecture for image encoding. Default is 'vgg16'.
            in_image_size (tuple): Input image size. Default is (3, 224, 224).
            margin_expansion_factor (int): Margin expansion factor for image encoding. Default is 6.
            primal_mp_layer_inputs (list): List of input sizes for primal message passing layers. Default is [728, 600, 512, 400, 256].
            dual_mp_layer_inputs (list): List of input sizes for dual message passing layers. Default is [1000, 728, 600, 512, 400].
            delimiter (str): Delimiter used for creating dual node indices. Default is "_".
            **kwargs: Additional keyword arguments for image encoder and dual message passing layers.
        """
        super(DualGATImageClustering, self).__init__()
        
        # Extract additional keyword arguments
        image_encoder_args = kwargs.get("image_encoder_args", {})
        dual_message_passing_args = kwargs.get("dual_message_passing_args", {})
        criterion_args = kwargs.get("criterion_args", {})

        # Store input size and primal index
        self.image_size = in_image_size
        self.primal_index = primal_index 
        self.dual_index = dual_index
        self.delimiter = delimiter 
        self.dec_primal_mp_layer_inputs = primal_mp_layer_inputs[::-1]
        self.dec_dual_mp_layer_inputs = dual_mp_layer_inputs[::-1]
        self.criterion = criterion
        self.primal_criterion_weights = primal_criterion_weights
        self.dual_criterion_weights = dual_criterion_weights

        self.image_encoder = ImageEncoder(input_shape=in_image_size, 
                                          model=backbone, 
                                          margin_expansion_factor=margin_expansion_factor, 
                                          **image_encoder_args)
        
        self.out_img_size = utils.get_output_model(self.image_encoder, in_image_size)
        

        self.enc_primal_mp_layer_inputs = [self.out_img_size] + primal_mp_layer_inputs
        self.enc_dual_mp_layer_inputs = [n_objects] + dual_mp_layer_inputs
        self.dual_depths = [n_objects] + primal_mp_layer_inputs

        # Create encoder and decoder message passing layers
        self.enc_dmp_layers = []
        self.dec_dmp_layers = []

        for i in range(len(self.enc_primal_mp_layer_inputs)-1):
            self.enc_dmp_layers.append(DualMessagePassing(primal_in_features=self.enc_primal_mp_layer_inputs[i], 
                                                      primal_out_features=self.enc_primal_mp_layer_inputs[i+1], 
                                                      primal_index=self.primal_index,
                                                      primal_depth=self.enc_dual_mp_layer_inputs[i],
                                                    
                                                      dual_in_features=self.enc_dual_mp_layer_inputs[i],
                                                      dual_out_features=self.enc_dual_mp_layer_inputs[i+1],
                                                      dual_index=self.dual_index,
                                                      dual_depth=self.dual_depths[i],
                                                      layer_index=f"encoder_{i}",
                                                      delimiter=self.delimiter,
                                                      **dual_message_passing_args))
            
        for i in range(len(self.dec_primal_mp_layer_inputs)-1):
            self.dec_dmp_layers.append(DualMessagePassing(primal_in_features=self.dec_primal_mp_layer_inputs[i], 
                                                      primal_out_features=self.dec_primal_mp_layer_inputs[i+1], 
                                                      primal_index=self.primal_index,
                                                      primal_depth=self.dec_dual_mp_layer_inputs[i],
                                                    
                                                      dual_in_features=self.dec_dual_mp_layer_inputs[i],
                                                      dual_out_features=self.dec_dual_mp_layer_inputs[i+1],
                                                      dual_index=self.dual_index,
                                                      dual_depth=self.dec_primal_mp_layer_inputs[i],
                                                      layer_index=f"decoder_{i}",
                                                      delimiter=self.delimiter,
                                                      **dual_message_passing_args))

    def encoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        encoder_history = {}
        for layer in self.enc_dmp_layers:
            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)
            
            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]

            encoder_history.extend(result)
        
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, encoder_history

    def decoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        decoder_history = {}
        for layer in self.dec_dmp_layers:
            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)
            
            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]
        
            decoder_history.extend(result)
            
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, decoder_history
        

    def forward(self, imgs, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        """
        Forward pass of the DualGATImageClustering model.
        
        Args:
            imgs (torch.Tensor): Input images.
            primal_adjacency_tensor (torch.Tensor): Primal adjacency tensor.
            dual_adjacency_tensor (torch.Tensor): Dual adjacency tensor.
            dual_nodes (torch.Tensor): Dual nodes.
        
        Returns:
            tuple: Tuple containing primal nodes, primal adjacency tensor, dual nodes, and dual adjacency tensor.
        """
        # Encode images to embeddings
        primal_nodes = self.image_encoder(imgs)

        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, encoder_history = self.encoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, decoder_history = self.decoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor
    
# # Create adjacency tensors and other variables
# mat1 = np.array([[1, 1, 0, 1, 1], 
#                 [1, 1, 1, 0, 0], 
#                 [0, 1, 1, 0, 1], 
#                 [1, 0, 0, 1, 0],
#                 [1, 0, 1, 0, 1]])

# mat2 = np.array([[1, 0, 1, 1, 1], 
#                 [0, 1, 1, 0, 1], 
#                 [1, 1, 1, 1, 1], 
#                 [1, 0, 1, 1, 0],
#                 [1, 1, 1, 0, 1]])

# mat3 = np.array([[1, 1, 0, 0, 1], 
#                 [1, 1, 0, 1, 0], 
#                 [0, 0, 1, 0, 1], 
#                 [0, 1, 0, 1, 0],
#                 [1, 0, 1, 0, 1]])

# mat4 = np.array([[1, 1, 0, 0, 1], 
#                 [1, 1, 0, 1, 0], 
#                 [0, 0, 1, 0, 1], 
#                 [0, 1, 0, 1, 0],
#                 [1, 0, 1, 0, 1]])

# primal_adjacency_tensor = torch.tensor(np.array([mat1, mat2, mat3, mat4]), dtype=constants.FLOATING_POINT)
# n_objects = primal_adjacency_tensor.shape[0]
# num_images = primal_adjacency_tensor.shape[1]

# img = torch.randn(num_images, 4, 224, 224)
# primal_index = ["1", "2", "3", "4", "5"]

# dual_index, dual_adjacency_tensor, dual_nodes = utils.create_dual_adjacency_tensor(primal_adjacency_tensor, primal_index, "_")

# # Create and run the DualGATImageClustering model
# model = DualGATImageClustering(primal_index=primal_index, dual_index=dual_index, n_objects=n_objects)
# result = model(img, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
# # print(result[3].shape)
