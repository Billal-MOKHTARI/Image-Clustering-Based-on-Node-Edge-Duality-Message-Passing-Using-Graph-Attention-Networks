from torch import nn
import constants
from dual_message_passing import DualMessagePassing
from image_encoder import ImageEncoder

class DualGATImageClustering(nn.Module):
    def __init__(self, 
                image_size, 
                backbone, 
                index,
                margin_expansion_factor=6,
                mp_layer_inputs=[728, 600, 512, 400, 256],  
                **kwargs):
        super(DualGATImageClustering, self).__init__()
        image_encoder_args = kwargs.get("image_encoder_args", {})
        
        self.image_size = image_size
        self.image_encoder = ImageEncoder(input_shape=image_size, 
                                          model=backbone, 
                                          margin_expansion_factor=margin_expansion_factor, 
                                          **image_encoder_args)
        self.index = index
        
    def create_dual_index(self):
        pass
        
    def forward(self, imgs):
        imgs = self.image_encoder(imgs)
        