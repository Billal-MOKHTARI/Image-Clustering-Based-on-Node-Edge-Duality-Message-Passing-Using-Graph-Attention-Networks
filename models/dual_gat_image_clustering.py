import torch
from torch import nn
from . import constants

from .dual_message_passing import DualMessagePassing
from .image_encoder import ImageEncoder
import numpy as np
import os
import sys
from src import utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


class DualGATImageClustering(nn.Module):
    def __init__(self, 
                primal_index,
                dual_index,
                n_objects,
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
        self.primal_mp_layer_inputs = [self.out_img_size] + primal_mp_layer_inputs

        # Create dual layer inputs
        self.dual_mp_layer_inputs = [n_objects] + dual_mp_layer_inputs
              
        # Create dual message passing depths
        self.dual_depths = [n_objects] + primal_mp_layer_inputs
        
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

    def encoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):

        for layer in self.dmp_layers:
            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)
            
            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]
        
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor

    def decoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        
        
        pass

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

        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor = self.encoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        self.decoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor