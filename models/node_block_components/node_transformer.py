import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../")
import constants

class NodeTransformer(nn.Module):
    def __init__(self, input_shape, backbone):
        super(NodeTransformer, self).__init__()

        assert backbone in constants.MODELS, f"The models you chosed is not in {constants.MODELS}"
    
    

        # The input images are tensors of W x H x C, where C >= 3
        self.input_shape = input_shape
        
