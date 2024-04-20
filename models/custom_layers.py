from torch import nn
import torch
import os
import sys
import numpy as np

import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import models
from torchsummary import summary

class Linear2D(nn.Module):

    def __init__(self, depth, n_features):
        """
        Initializes a Linear2D object.

        Args:
            depth (int): The number of linear layers to create.
            n_features (int): The number of input and output features for each linear layer.

        Example:
            >>> model = Linear2D(depth=3, n_features=5)

            >>> mat1 = np.array([[1, 1, 0, 1, 1], 
                            [1, 1, 1, 0, 0], 
                            [0, 1, 1, 0, 1], 
                            [1, 0, 0, 1, 0],
                            [1, 0, 1, 0, 1]])

            >>> mat2 = np.array([[1, 0, 1, 1, 1], 
                            [0, 1, 1, 0, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 0, 1, 1, 0],
                            [1, 1, 1, 0, 1]])

            >>> mat3 = np.array([[1, 1, 0, 0, 1], 
                            [1, 1, 0, 1, 0], 
                            [0, 0, 1, 0, 1], 
                            [0, 1, 0, 1, 0],
                            [1, 0, 1, 0, 1]])

            >>> mat = torch.tensor(np.array([mat1, mat2, mat3], dtype=np.float32))
        """
        super(Linear2D, self).__init__()
        self.list_linear = []
        self.depth = depth

        for i in range(depth):
            self.list_linear.append(nn.Linear(n_features, n_features))
   

    def forward(self, x):
        assert len(x.shape) == 3, "The input tensor should be a 3D tensor"
        assert len(x) == self.depth, "The depth of the tensor should be equal to the depth of the model"



        list_product = []
        for channel, layer in zip(x, self.list_linear):
            list_product.append(layer(channel))
            
        return torch.stack(list_product)
    
class Encoder2D(nn.Module):
    def __init__(self, 
                 layers, 
                 latent_dims, 
                 activation, 
                 num_conv_layers, 
                 k_sizes, 
                 strides, 
                 k_size_pool,
                 stride_pool,
                 pool_type,
                 dropout_p=0.5,
                 **kwargs):
        super(Encoder2D, self).__init__()
        self.activation_args = kwargs.get("activation_args", {})
        self.activation = activation(**self.activation_args)
        self.encoder_layers = nn.Sequential()
        
        for i in range(len(layers)-1):
            self.encoder_layers.append(self.conv2d_block(layers[i], 
                                                  layers[i+1], 
                                                  num_conv_layers[i], 
                                                  k_sizes[i], 
                                                  strides[i], 
                                                  k_size_pool[i] if type(k_size_pool) == list else k_size_pool,
                                                  stride_pool[i] if type(stride_pool) == list else stride_pool,
                                                  pool_type=pool_type[i] if type(pool_type) == list else pool_type
                                                  ))
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.LazyLinear(latent_dims[0]))
        
        
        length = len(latent_dims)
        for i in range(1, length-1):
            self.linear.append(self.mlp_block(latent_dims[i-1], latent_dims[i], dropout_p))
        
        if length > 1:
            self.linear.append(nn.Linear(latent_dims[-2], latent_dims[-1]))
            
    def mlp_block(self, in_features, out_features, dropout_p):
        mlp_layers = nn.Sequential()
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.BatchNorm1d(out_features))
        mlp_layers.append(self.activation)
        mlp_layers.append(nn.Dropout(dropout_p))
        
        return mlp_layers
    
    def conv2d_block(self, 
              in_channels, 
              out_channels, 
              num_conv_layers, 
              kernel_size, 
              stride, 
              pool_kernel=(2, 2), 
              pool_stride=(2, 2), 
              pool_type='max'):
        
        conv2d_layers = nn.Sequential()
        conv2d_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same'))
        conv2d_layers.append(nn.BatchNorm2d(out_channels)),
        conv2d_layers.append(self.activation)
        
        for _ in range(num_conv_layers-1):
            conv2d_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding='same'))
            conv2d_layers.append(nn.BatchNorm2d(out_channels)),
            conv2d_layers.append(self.activation)
        
        if pool_type == 'max':
            pool2d = nn.MaxPool2d(pool_kernel, pool_stride)
        elif pool_type == 'avg':
            pool2d = nn.AvgPool2d(pool_kernel, pool_stride)
           
           
        return nn.Sequential(
            conv2d_layers,
            pool2d,
        )
    
    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x
    
model = Encoder2D(layers=[3, 64, 128, 256, 512], 
                  latent_dims=[4096, 1000, 512], 
                  activation=nn.ReLU, 
                  num_conv_layers=[2, 2, 2, 2], 
                  k_sizes=[3, 3, 3, 3], 
                  strides=[1, 1, 1, 1], 
                  k_size_pool=2,
                  stride_pool=2,
                  pool_type='max')
# summary(model, (3, 224, 224))
x = torch.randn(2, 3, 224, 224)

output = model(x)
print(output.shape)
# summary(models.vgg16(pretrained=True), (3, 224, 224))