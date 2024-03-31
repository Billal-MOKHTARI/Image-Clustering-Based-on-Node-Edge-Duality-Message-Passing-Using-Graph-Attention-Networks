import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # Used for model summary
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import constants  # Import custom constants module
from src import utils  # Import custom utility functions module
from src import py_model_manager  # Import custom PyModelManager module

class NodeEncoder(nn.Module):
    """
    A custom neural network module for transforming node data using extended convolutional layers 
    followed by a pre-trained model.

    Args:
        input_shape (tuple): Input shape of the data (channels, height, width).
        model (str): Pre-trained model architecture name.
        margin_expansion_factor (int, optional): Factor for expanding margins (default is 6).
        **kwargs: Additional keyword arguments for model configuration.

    Attributes:
        input_shape (tuple): Input shape of the data.
        model (torch.nn.Module): Pre-trained model instance.
        margin_expansion_factor (int): Factor for expanding margins.
        extended_conv_layers (torch.nn.Sequential): Sequential container for extended convolutional layers.
    """

    def __init__(self, input_shape, model_name, margin_expansion_factor=6, **kwargs):
        super(NodeEncoder, self).__init__()
  
        # Ensure that the chosen model is supported
        assert model_name in constants.MODELS, f"The model you chose is not in {constants.MODELS}"

        # Extract additional keyword arguments
        self.model_name = model_name  # Pre-trained model architecture name
        pretrained = kwargs.get("pretrained", True)  # Whether to use pre-trained weights
        stride = kwargs.get("stride", (1, 1))  # Stride for convolutional layers
        padding = kwargs.get("padding", (0, 0))  # Padding for convolutional layers

        # Store input shape and pre-trained model
        self.input_shape = input_shape
        self.model = eval("models." + self.model_name)(pretrained=pretrained)  # Instantiate the pre-trained model
        
        self.py_model_manager = py_model_manager.PyModelManager(self.model)

        # if self.model_name.startswith('vgg') :
        #     self.py_model_manager.delete_layer_by_name('classifier', -1)
        
        self.margin_expansion_factor = margin_expansion_factor  # Margin expansion factor
        self.extended_conv_layers = list()  # List to store extended convolutional layers

        # Calculate kernel sizes for convolutional layers
        w_kernel = utils.required_kernel(self.input_shape[1], self.input_shape[1], stride[0], padding[0])
        h_kernel = utils.required_kernel(self.input_shape[2], self.input_shape[2], stride[1], padding[1])

        out_channels = self.input_shape[0] // 2  # Initialize output channels for extended convolutional layers
        tmp = self.input_shape[0]  # Initialize temporary variable for output channels

        # Create extended convolutional layers
        while out_channels > 3 * self.margin_expansion_factor:
            self.extended_conv_layers.append(nn.Conv2d(tmp,
                                                        out_channels=out_channels,
                                                        kernel_size=(w_kernel, h_kernel),
                                                        stride=stride,
                                                        padding=padding))
            tmp = out_channels  # Update temporary variable
            out_channels //= 2  # Decrease output channels by a factor of 2

        # Add final convolutional layer with 3 output channels
        self.extended_conv_layers.append(nn.Conv2d(tmp,
                                                    out_channels=3,
                                                    kernel_size=(w_kernel, h_kernel),
                                                    stride=stride,
                                                    padding=padding))

        self.extended_conv_layers = nn.Sequential(*self.extended_conv_layers)  # Convert to sequential container

    def forward(self, x):
        """
        Forward pass method of the NodeEncoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the extended convolutional layers and pre-trained model.
        """
        x = self.extended_conv_layers(x)  # Pass input through extended convolutional layers
        x = self.model(x)  # Pass input through pre-trained model
        return x



# # model.delete_layer_by_name('classifier', -1)
# print(model.get_named_layers())
# print(vgg.classifier[0])

input_shape = (200, 256, 256)
model = NodeEncoder(input_shape, model_name='efficientnet_b0')

pmm = py_model_manager.PyModelManager(model)
print(model.model.named_children())
# summary(model, input_shape, -1)