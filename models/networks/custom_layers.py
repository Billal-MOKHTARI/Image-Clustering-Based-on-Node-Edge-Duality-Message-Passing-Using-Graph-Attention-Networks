from torch import nn
import torch
import os
import sys
import math

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import maths
from src import files_manager

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
        self.layers = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(depth)])

    def forward(self, x):
        assert len(x.shape) == 3, "The input tensor should be a 3D tensor"
        assert len(x) == len(self.layers), "The depth of the tensor should be equal to the depth of the model"

        outputs = []
        for layer, input_channel in zip(self.layers, x):
            output_channel = layer(input_channel)
            outputs.append(output_channel)

        return torch.stack(outputs)

class Encoder2D(nn.Module):
    """
    2D Encoder module for convolutional autoencoder.
    """

    def __init__(self, 
                 shapes, 
                 latent_dims, 
                 activation, 
                 conv_layers_per_block, 
                 conv_kernels, 
                 conv_strides, 
                 pool_strides,
                 pool_paddings,
                 pool_types,
                 dropout_prob=0.5,
                 **kwargs):
        """
        Initialize the Encoder2D.

        Args:
            shapes (list): List of tuples specifying the shapes of input and output layers.
            latent_dims (list): List of dimensions of latent space.
            activation (torch.nn.Module): Activation function.
            conv_layers_per_block (list or int): Number of convolutional layers.
            conv_kernels (list or int): Kernel size for convolutional layers.
            conv_strides (list or int): Stride for convolutional layers.
            pool_strides (list or int): Stride for pooling layers.
            pool_paddings (list or int): Padding for pooling layers.
            pool_types (str or list): Pooling type, either 'max' or 'avg'.
            dropout_prob (float): Dropout probability.
            **kwargs: Additional keyword arguments.
        """
        super(Encoder2D, self).__init__()
        self.activation_args = kwargs.get("activation_args", {})
        self.activation = activation(**self.activation_args)
        self.encoder_layers = []
        layers = [layer[0] for layer in shapes]
        sizes = [(layer[1], layer[2]) for layer in shapes]
        
        # Create convolutional layers
        for i in range(len(layers)-1):
            self.encoder_layers.extend(self.conv2d_block(layers[i], 
                                                         layers[i+1], 
                                                         conv_layers_per_block[i] if type(conv_layers_per_block) == list else conv_layers_per_block, 
                                                         conv_kernels[i] if type(conv_kernels) == list else conv_kernels, 
                                                         conv_strides[i] if type(conv_strides) == list else conv_strides, 
                                                         (maths.required_kernel(sizes[i][0], sizes[i+1][0], pool_strides[i][0] if type(pool_strides) == list else pool_strides[0], pool_paddings[i][0] if type(pool_paddings) == list else pool_paddings[0]),
                                                          maths.required_kernel(sizes[i][1], sizes[i+1][1], pool_strides[i][1] if type(pool_strides) == list else pool_strides[1], pool_paddings[i][1] if type(pool_paddings) == list else pool_paddings[1])), 
                                                         pool_strides[i] if type(pool_strides) == list else pool_strides,
                                                         pool_paddings[i] if type(pool_paddings) == list else pool_paddings,
                                                         pool_types=pool_types[i] if type(pool_types) == list else pool_types
                                                         ))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.LazyLinear(latent_dims[0]))
        
        # Create linear layers
        length = len(latent_dims)
        for i in range(1, length-1):
            self.linear.append(self.mlp_block(latent_dims[i-1], latent_dims[i], dropout_prob))
        
        if length > 1:
            self.linear.append(nn.Linear(latent_dims[-2], latent_dims[-1]))
            

    def mlp_block(self, in_features, out_features, dropout_prob):
        """
        Create a block of fully connected layers.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            dropout_prob (float): Dropout probability.

        Returns:
            torch.nn.Sequential: Sequential container for the fully connected layers.
        """
        mlp_layers = nn.Sequential()
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.BatchNorm1d(out_features))
        mlp_layers.append(self.activation)
        mlp_layers.append(nn.Dropout(dropout_prob))
        
        return mlp_layers
    
    def conv2d_block(self, 
                     in_channels, 
                     out_channels, 
                     conv_layers_per_block, 
                     kernel_size, 
                     conv_stride, 
                     pool_kernel, 
                     pool_stride=(2, 2), 
                     pool_padding=(0, 0),
                     pool_types='max'):
        """
        Create a block of convolutional layers followed by pooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            conv_layers_per_block (int): Number of convolutional layers.
            kernel_size (int or tuple): Size of the convolutional kernel.
            conv_stride (int or tuple): Stride of the convolutional operation.
            pool_kernel (int or tuple): Size of the pooling kernel.
            pool_stride (int or tuple): Stride of the pooling operation.
            pool_padding (int or tuple): Padding of the pooling operation.
            pool_types (str): Type of pooling operation, either 'max' or 'avg'.

        Returns:
            torch.nn.Sequential: Sequential container for the convolutional and pooling layers.
        """
        conv2d_layers = []
        conv2d_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, conv_stride, padding='same'))
        conv2d_layers.append(nn.BatchNorm2d(out_channels))
        conv2d_layers.append(self.activation)
        
        for _ in range(conv_layers_per_block-1):
            conv2d_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, conv_stride, padding='same'))
            conv2d_layers.append(nn.BatchNorm2d(out_channels))
            conv2d_layers.append(self.activation)
        
        if pool_types == 'max':
            pool2d = nn.MaxPool2d(pool_kernel, pool_stride, pool_padding, return_indices=False)
        elif pool_types == 'avg':
            pool2d = nn.AvgPool2d(pool_kernel, pool_stride, pool_padding, return_indices=False)
        
        conv2d_layers.append(pool2d)
           
        return conv2d_layers
    
    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after encoding.
            list: List of pooling indices for each pooling layer.
        """
        pool_indices = []
        conv_encoder_history = []
        conv_encoder_history.append(x)
        
        for i in range(len(self.encoder_layers)):
            
            if isinstance(self.encoder_layers[i], nn.MaxPool2d):
                x, indices = self.encoder_layers[i](x)
                print(indices.shape)
                pool_indices.append(indices)
                conv_encoder_history.append(x)
            else:
                x = self.encoder_layers[i](x)

        x = self.flatten(x)
        x = self.linear(x)
        
        return x, pool_indices, conv_encoder_history
    
class Decoder2D(nn.Module):
    """
    2D Decoder module for convolutional autoencoder.
    """
    
    def __init__(self, 
                 latent_dims, 
                 shapes, 
                 activation, 
                 deconv_layers_per_block, 
                 deconv_strides, 
                 deconv_paddings,
                 unpool_strides,
                 unpool_paddings,
                 dropout_prob=0.5,
                 **kwargs):
        """
        Initialize the Decoder2D.

        Args:
            latent_dims (list): List of dimensions of latent space.
            shapes (list): List of tuples specifying the shapes of input and output layers.
            activation (torch.nn.Module): Activation function.
            conv_layers_per_block (list or int): Number of convolutional layers.
            conv_strides (list or int): Stride for convolutional layers.
            padding (list or int): Padding for convolutional layers.
            pool_strides (list or int): Stride for unpooling layers.
            pool_paddings (list or int): Padding for unpooling layers.
            dropout_prob (float): Dropout probability.
            **kwargs: Additional keyword arguments.
        """
        super(Decoder2D, self).__init__()
        self.activation_args = kwargs.get("activation_args", {})
        self.activation = activation(**self.activation_args)
        self.mlp_layers = nn.Sequential()
        self.decoder_layers = []
        layers = [layer[0] for layer in shapes]
        sizes = [(layer[1], layer[2]) for layer in shapes]
        
        # Create linear layers
        length = len(latent_dims)
        for i in range(length-2):
            self.mlp_layers.append(self.mlp_block(latent_dims[i], latent_dims[i+1], dropout_prob))
            
        if length > 1:
            self.mlp_layers.append(nn.Linear(latent_dims[-2], latent_dims[-1]))
        
        # Create convolutional layers
        kernel_size = (maths.required_kernel_transpose(1, sizes[0][0], stride=1, padding=0),
                          maths.required_kernel_transpose(1, sizes[0][1], stride=1, padding=0))
        self.decoder_layers.append(nn.ConvTranspose2d(latent_dims[-1], layers[0], kernel_size, 1, padding=0))
        
        for i in range(len(layers)-1):
            self.decoder_layers.extend(self.deconv2d_block(layers[i], 
                                                           layers[i+1], 
                                                           deconv_layers_per_block[i] if type(deconv_layers_per_block) == list else deconv_layers_per_block, 
                                                           deconv_stride = deconv_strides[i] if type(deconv_strides) == list else deconv_strides, 
                                                           deconv_padding = deconv_paddings[i] if type(deconv_paddings) == list else deconv_paddings,
                                                           in_size=sizes[i],
                                                           out_size=sizes[i+1],
                                                           unpool_stride = unpool_strides[i] if type(unpool_strides) == list else unpool_strides,
                                                           unpool_padding = unpool_paddings[i] if type(unpool_paddings) == list else unpool_paddings   
                                                           ))
        

        
    def mlp_block(self, in_features, out_features, dropout_prob):
        """
        Create a block of fully connected layers.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            dropout_prob (float): Dropout probability.

        Returns:
            torch.nn.Sequential: Sequential container for the fully connected layers.
        """
        mlp_layers = nn.Sequential()
        mlp_layers.append(nn.Linear(in_features, out_features))
        mlp_layers.append(nn.BatchNorm1d(out_features))
        mlp_layers.append(self.activation)
        mlp_layers.append(nn.Dropout(dropout_prob))
        
        return mlp_layers
    
    def deconv2d_block(self, 
                       in_channels, 
                       out_channels, 
                       deconv_layers_per_block, 
                       deconv_stride, 
                       deconv_padding,
                       in_size,
                       out_size,
                       unpool_stride=(2, 2),
                       unpool_padding=(0, 0)):
        """
        Create a block of transposed convolutional layers followed by unpooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            conv_layers_per_block (int): Number of convolutional layers.
            stride (int or tuple): Stride for convolutional layers.
            padding (int or tuple): Padding for convolutional layers.
            in_size (tuple): Size of the input feature map.
            out_size (tuple): Size of the output feature map.
            pool_stride (int or tuple): Stride for unpooling layers.
            pool_padding (int or tuple): Padding for unpooling layers.

        Returns:
            torch.nn.Sequential: Sequential container for the transposed convolutional and unpooling layers.
        """
        deconv2d_layers = []
        pool_kernel_size = (maths.required_kernel_transpose(in_size[0], out_size[0], stride=unpool_stride[0], padding=unpool_padding[0]),
                            maths.required_kernel_transpose(in_size[1], out_size[1], stride=unpool_stride[1], padding=unpool_padding[1]))
        unpool2d = nn.MaxUnpool2d(pool_kernel_size, unpool_stride, unpool_padding)
        deconv2d_layers.append(unpool2d)

        # We should keep the input size as it is
        deconv_kernel_size = (maths.required_kernel_transpose(in_size[0], in_size[0], stride=deconv_stride[0], padding=deconv_padding[0]),
                       maths.required_kernel_transpose(in_size[1], in_size[1], stride=deconv_stride[1], padding=deconv_padding[1]))
        
        deconv2d_layers.append(nn.ConvTranspose2d(in_channels, out_channels, deconv_kernel_size, deconv_stride, padding=deconv_padding))
        deconv2d_layers.append(nn.BatchNorm2d(out_channels))
        deconv2d_layers.append(self.activation)
        
        for _ in range(deconv_layers_per_block-1):
            deconv2d_layers.append(nn.ConvTranspose2d(out_channels, out_channels, deconv_kernel_size, deconv_stride, deconv_padding))
            deconv2d_layers.append(nn.BatchNorm2d(out_channels))
            deconv2d_layers.append(self.activation)
        
        return deconv2d_layers
    
    def forward(self, x, pool_indices):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor.
            pool_indices (list): List of pooling indices for each pooling layer.

        Returns:
            torch.Tensor: Output tensor after decoding.
        """
        x = self.mlp_layers(x)
        x = x.view(x.size(0), -1, 1, 1)  # Reshape to (batch_size, channels, height, width)
        deconv_decoder_history = []
        k = 0 
        for i in range(len(self.decoder_layers)):
            if isinstance(self.decoder_layers[i], nn.MaxUnpool2d):
                deconv_decoder_history.append(x)
                
                x = self.decoder_layers[i](x, pool_indices[k])
                k += 1
            else:
                x = self.decoder_layers[i](x)
                
        deconv_decoder_history.append(x)
        
        return x, deconv_decoder_history

class AutoEncoder(nn.Module):
    def __init__(self, encoder_args, decoder_args, loss, **kwargs):
        super(AutoEncoder, self).__init__()
        loss_args = kwargs.get("loss_args", {})
        self.encoder = Encoder2D(**encoder_args)
        self.decoder = Decoder2D(**decoder_args)
        self.loss = loss(**loss_args)
        
    def forward(self, x):
        x, pool_indices, conv_encoder_history = self.encoder(x)
        print(conv_encoder_history)
        x, deconv_decoder_history = self.decoder(x, pool_indices)
        print(deconv_decoder_history)
        
        
        return x, conv_encoder_history, deconv_decoder_history

# Specify the path to your JSON file
# encoder_json_file_path = "configs/rough/encoder.json"
# decoder_json_file_path = "configs/rough/decoder.json"

# encoder_args = files_manager.parse_encoder(encoder_json_file_path, network_type="encoder")
# decoder_args = files_manager.parse_encoder(decoder_json_file_path, network_type="decoder")

# auto_encoder = AutoEncoder(encoder_args, decoder_args, nn.MSELoss)

# x_enc = torch.randn(2, 1, 244, 224)
# x_dec = torch.randn(2, 128)

# auto_encoder(x_enc)
# # for enc in conv_encoder_history:
# #     print(enc.shape)

# decoder = Decoder2D(**decoder_args)
# y_dec, deconv_decoder_history = decoder(y_enc, indices[::-1])

# for dec in deconv_decoder_history:
#     print(dec.shape)
