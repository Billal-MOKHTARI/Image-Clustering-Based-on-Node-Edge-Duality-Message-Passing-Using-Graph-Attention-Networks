import torch
from torch import nn
from torch import optim
from dual_message_passing import DualMessagePassing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src import utils

import pandas as pd
from torch import nn
import constants
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from custom_layers import Encoder2D, Decoder2D, parse_encoder

def read_images(folder_path, n):
    """
    Read the first n images from a folder, resize them to (3, 224, 224), and stack them into a single tensor.

    Args:
        folder_path (str): Path to the folder containing images.
        n (int): Number of images to read.

    Returns:
        torch.Tensor: Stacked tensor of shape (N, 3, 224, 224).
    """
    images = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    for filename in sorted(os.listdir(folder_path)):
        if len(images) == n:
            break
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                img_tensor = transform(img)
                images.append(img_tensor)
    stacked_tensor = torch.stack(images, dim=0)
    return stacked_tensor

class DualGATImageClustering(nn.Module):
    def __init__(self, 
                primal_index,
                dual_index,
                n_objects,
                criterion = nn.MSELoss,
                primal_criterion_weights = [1, 1, 1, 1, 1],
                dual_criterion_weights = [1, 1, 1, 1, 1],
                primal_mp_layer_inputs=[64, 32, 16, 8, 4],  
                dual_mp_layer_inputs=[64, 32, 16, 8],
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
        dual_message_passing_args = kwargs.get("dual_message_passing_args", {})
        criterion_args = kwargs.get("criterion_args", {})
        self.encoder_args = kwargs.get("encoder_args", {})
        self.decoder_args = kwargs.get("decoder_args", {})
        
        # Store input size and primal index
        self.primal_index = primal_index 
        self.dual_index = dual_index
        self.delimiter = delimiter 
        self.criterion = criterion(**criterion_args)
        self.primal_criterion_weights = primal_criterion_weights
        self.dual_criterion_weights = dual_criterion_weights

        # Image encoder parameters

        self.enc_primal_mp_layer_inputs = primal_mp_layer_inputs
        self.enc_dual_mp_layer_inputs = [n_objects] + dual_mp_layer_inputs
        self.dec_primal_mp_layer_inputs = self.enc_primal_mp_layer_inputs[::-1]
        self.dec_dual_mp_layer_inputs = self.enc_dual_mp_layer_inputs[::-1]


        self.primal_depths = [n_objects] + dual_mp_layer_inputs
        self.dual_depths = [n_objects] + primal_mp_layer_inputs[1:]

        # Create encoder and decoder message passing layers
        self.enc_dmp_layers = []
        self.dec_dmp_layers = []

        self.image_encoder = self.get_image_encoder()
        self.image_decoder = self.get_image_decoder()

        for i in range(len(self.enc_primal_mp_layer_inputs)-1):
            self.enc_dmp_layers.append(DualMessagePassing(primal_in_features=self.enc_primal_mp_layer_inputs[i], 
                                                      primal_out_features=self.enc_primal_mp_layer_inputs[i+1], 
                                                      primal_index=self.primal_index,
                                                      primal_depth=self.primal_depths[i],
                                                    
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

    def get_image_encoder(self):

        return Encoder2D(**self.encoder_args)


    def get_image_decoder(self):
        return Decoder2D(**self.decoder_args)


    def encoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        encoder_history = []
 
        for layer in self.enc_dmp_layers:
            tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
            encoder_history.append(tmp)

            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)
            
            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]

        tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
        encoder_history.append(tmp)
        
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, encoder_history

    def decoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):

        decoder_history = []

        for layer in self.dec_dmp_layers:

            tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
            decoder_history.append(tmp)

            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)

            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]
        

        tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
        decoder_history.append(tmp)


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
        
        encoder_history = []
        
        primal_nodes, indices, conv_encoder_history = self.image_encoder(imgs)
        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, encoder_history = self.encoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, decoder_history = self.decoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        
        print(primal_nodes.shape)
        primal_nodes, deconv_decoder_history = self.image_decoder(primal_nodes, indices[::-1])
        
        decoder_history = decoder_history[::-1]

        encoder_history.pop()
        decoder_history.pop()

        # Compute primal reconstruction loss
        primal_losses = []
        dual_losses = []

        num_layers = len(encoder_history)
        for i in range(num_layers):
            enc_primal_node = encoder_history[i]["primal_nodes"]
            dec_primal_node = decoder_history[i]["primal_nodes"]
            primal_losses.append(self.primal_criterion_weights[i] * self.criterion(enc_primal_node, dec_primal_node))

            enc_dual_node = encoder_history[i]["dual_nodes"]
            dec_dual_node = decoder_history[i]["dual_nodes"]
            dual_losses.append(self.dual_criterion_weights[i] * self.criterion(enc_dual_node, dec_dual_node))

        # print(self.image_decoder(primal_nodes))
        result = {
            "primal": {
                "nodes": primal_nodes,
                "adjacency_tensor": primal_adjacency_tensor,
                "losses": primal_losses},
            "dual": {
                "nodes": dual_nodes,
                "adjacency_tensor": dual_adjacency_tensor,
                "losses": dual_losses
            },
            "conv_encoder_history": conv_encoder_history,
            "deconv_decoder_history": deconv_decoder_history[::-1]
        }

        return result

def train(model, 
          images, 
          epochs, 
          optimizer = optim.Adam, 
          primal_weight = 1, 
          dual_weight = 1, 
          **kwargs):
    
    assert primal_weight >= 0 and primal_weight <= 1, "Primal weight must be between 0 and 1"
    assert dual_weight >= 0 and dual_weight <= 1, "Dual weight must be between 0 and 1"
    
    optimizer = optimizer(model.parameters(), **kwargs)
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        result = model(images, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        primal_loss = utils.list_sum(result["primal"]["losses"])
        dual_loss = utils.list_sum(result["dual"]["losses"])
        conv_encoder_history = result["conv_encoder_history"]
        deconv_decoder_history = result["deconv_decoder_history"]
        
        print("encoder")
        for enc in conv_encoder_history:
            print(enc.shape)
        
        print("decoder")
        for dec in deconv_decoder_history:
            print(dec.shape)
        loss = (primal_weight*primal_loss + dual_weight*dual_loss)/(primal_weight+dual_weight)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


# Test the model
# Define the adjacency matrix
n = 3  # Number of images to read and convert

mat_square = pd.read_csv("../benchmark/datasets/test/adjacency_matrix_square.csv", index_col=0, header=0, dtype=np.float32)
mat_circle = pd.read_csv("../benchmark/datasets/test/adjacency_matrix_circle.csv", index_col=0, header=0, dtype=np.float32)
mat_triangle = pd.read_csv("../benchmark/datasets/test/adjacency_matrix_triangle.csv", index_col=0, header=0, dtype=np.float32)

primal_adjacency_tensor = torch.tensor(np.array([mat_square, mat_circle, mat_triangle]), dtype=constants.FLOATING_POINT)[:,:n,:n]
n_objects = primal_adjacency_tensor.shape[0]
num_images = primal_adjacency_tensor.shape[1]

primal_index = utils.convert_list(mat_square.index.tolist(), np.float32, str)[:n]
dual_index, dual_adjacency_tensor, dual_nodes = utils.create_dual_adjacency_tensor(primal_adjacency_tensor, 
                                                                                   primal_index, 
                                                                                   "_")


# Example usage:
folder_path = "/home/billalmokhtari/Documents/projects/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/benchmark/datasets/test/shapes"
img = read_images(folder_path, n)


encoder_json_file_path = "/home/billalmokhtari/Documents/projects/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/configs/encoder.json"
decoder_json_file_path = "/home/billalmokhtari/Documents/projects/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/configs/decoder.json"
encoder_args = parse_encoder(encoder_json_file_path, network_type="encoder")
decoder_args = parse_encoder(decoder_json_file_path, network_type="decoder")

# Create and run the DualGATImageClustering model
model = DualGATImageClustering(primal_index=primal_index, 
                               dual_index=dual_index, 
                               n_objects=n_objects, 
                               primal_criterion_weights=[1, 0.2, 0.3, 0.2, 1], 
                               dual_criterion_weights=[1, 0.2, 0.3, 0.2, 1],
                               image_size=(3, 64, 64),
                               encoder_args=encoder_args, decoder_args=decoder_args)

epochs = 100
train(model, img, epochs, optimizer=optim.Adam, lr=0.01)