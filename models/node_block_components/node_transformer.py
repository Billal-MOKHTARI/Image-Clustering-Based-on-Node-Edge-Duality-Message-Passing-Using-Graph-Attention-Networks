import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # Used for model summary

import sys
sys.path.append("../../")  # Add parent directory to system path
sys.path.append("../")  # Add current directory to system path
import constants  # Import custom constants module
from src import utils  # Import custom utility functions module

# Define a custom neural network module called NodeTransformer
class NodeTransformer(nn.Module):
    def __init__(self, 
                input_shape,  # Input shape of the data (tuple: (channels, height, width))
                model,  # Pre-trained model architecture name
                margin_expansion_factor=6,  # Factor for expanding margins
                **kwargs):  # Additional keyword arguments
        
        super(NodeTransformer, self).__init__()  # Call the constructor of the parent class
        
        # Ensure that the chosen model is supported
        assert model in constants.MODELS, f"The model you chose is not in {constants.MODELS}"
        
        # Extract additional keyword arguments
        pretrained = kwargs.get("pretrained", True)  # Whether to use pre-trained weights
        stride = kwargs.get("stride", (1, 1))  # Stride for convolutional layers
        padding = kwargs.get("padding", (0, 0))  # Padding for convolutional layers

        # Store input shape and pre-trained model
        self.input_shape = input_shape
        self.model = eval("models." + model)(pretrained=pretrained)  # Instantiate the pre-trained model
        self.margin_expansion_factor = margin_expansion_factor  # Margin expansion factor
        extended_conv_layers = list()  # List to store extended convolutional layers
        
        # Calculate kernel sizes for convolutional layers
        w_kernel = utils.required_kernel(self.input_shape[1], self.input_shape[1], stride[0], padding[0])
        h_kernel = utils.required_kernel(self.input_shape[2], self.input_shape[2], stride[1], padding[1])
        
        tmp = out_channels  # Initialize temporary variable for output channels
        out_channels = self.input_shape[0] // 2  # Initialize output channels for extended convolutional layers
        
        # Create extended convolutional layers
        while out_channels > 3 * self.margin_expansion_factor:
            extended_conv_layers.append(nn.Conv2d(tmp,
                                                  out_channels=out_channels,
                                                  kernel_size=(w_kernel, h_kernel),
                                                  stride=stride,
                                                  padding=padding))
            tmp = out_channels  # Update temporary variable
            out_channels //= 2  # Decrease output channels by a factor of 2
        
        # Add final convolutional layer with 3 output channels
        extended_conv_layers.append(nn.Conv2d(tmp,
                                              out_channels=3,
                                              kernel_size=(w_kernel, h_kernel),
                                              stride=stride,
                                              padding=padding))
        
    def get_model_summary(self):
        # Get a summary of the pre-trained model
        summary(self.model, input_size=self.model_input_size, batch_size=-1)
        
    def get_backbone_summary(self):
        pass  # Placeholder method for getting summary of backbone
        
    def forward(self, x):
        # Implement the forward pass of the NodeTransformer module
        # x is the input tensor of shape (batch_size, C, H, W)
        # Perform any necessary transformations or computations
        
        # Return the output tensor
        return x

        
# vgg16 = models.convnext_large()
# summary(vgg16, input_size = (3, 256, 256), batch_size = -1)