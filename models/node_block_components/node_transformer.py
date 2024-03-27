import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys

sys.path.append("../")
import constants

class NodeTransformer(nn.Module):
    def __init__(self, 
                input_shape, 
                model, 
                model_input_size=(3, 224, 244), 
                margin_expansion_factor=6, 
                **kwargs):
        
        super(NodeTransformer, self).__init__()

        assert model in constants.MODELS, f"The models you chosed is not in {constants.MODELS}"
        
        self.pretrained = kwargs.get("pretrained", True)
        self.model_input_size = model_input_size
        self.input_shape = input_shape
        self.model = eval("models." + model)(pretrained = self.pretrained)
        self.margin_expansion_factor = margin_expansion_factor
        
        extended_conv_layers = list()
        
        tmp = out_channels
        out_channels = self.input_shape[0]//2
        while out_channels > self.margin_expansion_factor*model_input_size[0]:
            extended_conv_layers.append(nn.Conv2d(tmp, out_channels=out_channels))
            
            tmp = out_channels
            out_channels //= 2
            
        
        
    def get_model_summary(self):
        summary(self.model, input_size = self.model_input_size, batch_size = -1)
        
    def get_backbone_summary(self):
        pass
    
    def forward(self, x):
        # Implement the forward pass of the NodeTransformer module
        # x is the input tensor of shape (batch_size, C, H, W)
        # Perform any necessary transformations or computations
        
        # Return the output tensor
        return x
        
# vgg16 = models.convnext_large()
# summary(vgg16, input_size = (3, 256, 256), batch_size = -1)