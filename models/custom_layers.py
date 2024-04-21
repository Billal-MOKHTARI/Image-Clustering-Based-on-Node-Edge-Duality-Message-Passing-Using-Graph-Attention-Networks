from torch import nn
import torch
import os
import sys
import numpy as np

import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src import maths
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
                 # Instead of manually defining the kernel size, stride, and padding for each pooling layer,
                 # we can define the input and out sizes, then the kernel sizes will be calculated automatically.
                 # The padding and stride will be fixed
                 sizes,
                 stride_pool,
                 padding_pool,
                 pool_type,
                 dropout_p=0.5,
                 **kwargs):
        super(Encoder2D, self).__init__()
        self.activation_args = kwargs.get("activation_args", {})
        self.activation = activation(**self.activation_args)
        self.encoder_layers = []
        self.pool_indices = None
        
        for i in range(len(layers)-1):
            self.encoder_layers.extend(self.conv2d_block(layers[i], 
                                                  layers[i+1], 
                                                  num_conv_layers[i] if type(num_conv_layers) == list else num_conv_layers, 
                                                  k_sizes[i] if type(k_sizes) == list else k_sizes, 
                                                  strides[i] if type(strides) == list else strides, 
                                                  (maths.required_kernel(sizes[i][0], sizes[i+1][0], stride_pool[i] if type(stride_pool) == list else stride_pool, padding_pool[i] if type(padding_pool) == list else padding_pool),
                                                   maths.required_kernel(sizes[i][1], sizes[i+1][1], stride_pool[i] if type(stride_pool) == list else stride_pool, padding_pool[i] if type(padding_pool) == list else padding_pool)), 
                                                  stride_pool[i] if type(stride_pool) == list else stride_pool,
                                                  padding_pool[i] if type(padding_pool) == list else padding_pool,
                                                  pool_type=pool_type[i] if type(pool_type) == list else pool_type
                                                  ))
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
              pool_kernel, 
              pool_stride=(2, 2), 
              pool_padding=(0, 0),
              pool_type='max'):
        
        conv2d_layers = []
        conv2d_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same'))
        conv2d_layers.append(nn.BatchNorm2d(out_channels)),
        conv2d_layers.append(self.activation)
        
        for _ in range(num_conv_layers-1):
            conv2d_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding='same'))
            conv2d_layers.append(nn.BatchNorm2d(out_channels)),
            conv2d_layers.append(self.activation)
        
        if pool_type == 'max':
            pool2d = nn.MaxPool2d(pool_kernel, pool_stride, pool_padding, return_indices=True)
        elif pool_type == 'avg':
            pool2d = nn.AvgPool2d(pool_kernel, pool_stride, pool_padding, return_indices=True)
        
        conv2d_layers.append(pool2d)
           
        return conv2d_layers
    
    def forward(self, x):
        pool_indices = []
        for i in range(len(self.encoder_layers)):
            if isinstance(self.encoder_layers[i], nn.MaxPool2d):
                x, indices = self.encoder_layers[i](x)
                pool_indices.append(indices)
            else:
                x = self.encoder_layers[i](x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x, pool_indices
    
class Decoder2D(nn.Module):
    def __init__(self, 
                 latent_dims, 
                 layers, 
                 activation, 
                 num_conv_layers, 
                 k_sizes, 
                 strides, 
                 
                 k_size_pool,
                 stride_pool,
                
                 dropout_p=0.5,
                 **kwargs):
        super(Decoder2D, self).__init__()
        self.activation_args = kwargs.get("activation_args", {})
        self.activation = activation(**self.activation_args)
        self.mlp_layers = nn.Sequential()
        self.decoder_layers = []
        
        length = len(latent_dims)
        for i in range(length-2):
            self.mlp_layers.append(self.mlp_block(latent_dims[i], latent_dims[i+1], dropout_p))
            
        if length > 1:
            self.mlp_layers.append(nn.Linear(latent_dims[-2], latent_dims[-1]))

        self.decoder_layers.extend(self.deconv2d_block(latent_dims[-1], 
                                                layers[0], 
                                                num_conv_layers[0], 
                                                k_sizes[0], 
                                                strides[0], 
                                                k_size_pool[0] if type(k_size_pool) == list else k_size_pool,
                                                stride_pool[0] if type(stride_pool) == list else stride_pool   
                                                ))
        
        for i in range(len(layers)-1):
            self.decoder_layers.extend(self.deconv2d_block(layers[i], 
                                                           layers[i+1], 
                                                           num_conv_layers[i] if type(num_conv_layers) == list else num_conv_layers[i], 
                                                           k_sizes[i] if type(k_sizes) == list else k_sizes[i], 
                                                           strides[i] if type(strides) == list else strides[i], 
                                                           k_size_pool[i] if type(k_size_pool) == list else k_size_pool,
                                                           stride_pool[i] if type(stride_pool) == list else stride_pool   
                                                           ))
        

        
    def mlp_block(self, in_features, out_features, dropout_p):
        mlp_layers = nn.Sequential()
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.BatchNorm1d(out_features))
        mlp_layers.append(self.activation)
        mlp_layers.append(nn.Dropout(dropout_p))
        
        return mlp_layers
    
    def deconv2d_block(self, 
                       in_channels, 
                       out_channels, 
                       num_conv_layers, 
                       kernel_size, 
                       stride, 
                       
                       pool_kernel, 
                       pool_stride=(2, 2),
                       pool_padding=(0, 0)):
        
        deconv2d_layers = []

        
        deconv2d_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=0))
        deconv2d_layers.append(nn.BatchNorm2d(out_channels))
        deconv2d_layers.append(self.activation)
        
        for _ in range(num_conv_layers-1):
            deconv2d_layers.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding=0))
            deconv2d_layers.append(nn.BatchNorm2d(out_channels))
            deconv2d_layers.append(self.activation)
            
        unpool2d = nn.MaxUnpool2d(pool_kernel, pool_stride, pool_padding)
        deconv2d_layers.append(unpool2d)

        
        return deconv2d_layers
    
    def forward(self, x, pool_indices):
        x = self.mlp_layers(x)

        x = x.view(x.size(0), -1, 1, 1)  # Reshape to (batch_size, channels, height, width)
        
        for i in range(len(self.decoder_layers)):
            k = 0
            if isinstance(self.decoder_layers[i], nn.MaxUnpool2d):
                x = self.decoder_layers[i](x, pool_indices[k])
                k += 1
            else:
                x = self.decoder_layers[i](x)
      
        
        
        return x

layers = [3, 64, 128, 256, 512]
latent_dims = [4096, 1000, 512]

sizes = [(224, 224), (112, 112), (56, 56), (28, 28), (14, 14)]
padding = 0
stride = 2

encoder = Encoder2D(layers=layers, 
                  latent_dims=latent_dims, 
                  activation=nn.ReLU, 
                  num_conv_layers=2, 
                  k_sizes=3, 
                  strides=1, 
                  sizes=sizes,
                  stride_pool=stride,
                  padding_pool=padding,
                  pool_type='max')

in_sizes = [14, 28, 56, 112]
out_sizes = [28, 56, 112, 224]
padding = 0
stride = 2
conv_transpose_kernels = [maths.required_kernel_transpose(in_size, out_size, stride=stride, padding=padding) for in_size, out_size in zip(in_sizes, out_sizes)]


decoder = Decoder2D(layers=layers[::-1], 
                  latent_dims=latent_dims[::-1], 
                  activation=nn.ReLU, 
                  num_conv_layers=[2, 2, 2, 2], 
                  k_sizes=[3, 3, 3, 3], 
                  strides=[1, 1, 1, 1], 
                  k_size_pool=conv_transpose_kernels,
                  stride_pool=stride,
                )

# summary(model, (3, 224, 224))
x_enc = torch.randn(2, 3, 224, 224)
x_dec = torch.randn(2, 512)
y_enc, indices = encoder(x_enc)
print(y_enc.shape)


# y_dec = decoder(y_enc, indices[::-1])
# print(y_dec)
# x = torch.randn(2, 4096, 1, 1)
# layer = nn.ConvTranspose2d(4096, 512, 3, 1) 
# out = layer(x)
# print(out.shape)