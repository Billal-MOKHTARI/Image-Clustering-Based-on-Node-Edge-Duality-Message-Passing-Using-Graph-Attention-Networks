from torch import nn
from torchsummary import summary
import torch
class IntermediateEncoder(nn.Module):
    def __init__(self, in_features, out_features, layer_index, **kwargs):
        super(IntermediateEncoder, self).__init__()
        linear_layer_args_name = "linear_layer"
        batch_norm_args_name = "batch_norm_layer"
        activation_args_name = "activation_layer"
        self.activation_exists = False
        self.layer_index = layer_index

        if linear_layer_args_name in kwargs.keys():
            linear_layer_args = kwargs[linear_layer_args_name]
        if batch_norm_args_name in kwargs.keys():
            batch_norm_args = kwargs[batch_norm_args_name]
        if activation_args_name in kwargs.keys():
            activation_name = "activation"
            assert activation_name in kwargs[activation_args_name], "The activation layer should be specified"
            assert issubclass(kwargs[activation_args_name][activation_name], nn.Module), "The activation layer should be a torch.nn.Module"
            self.activation = kwargs[activation_args_name][activation_name](**kwargs[activation_args_name]["activation_args"])
            self.activation_exists = True
            
        self.linear_layer = nn.Linear(in_features, out_features, **linear_layer_args)
        self.batch_norm = nn.BatchNorm1d(out_features, **batch_norm_args)
        
    def get_layer_index(self):
        return self.layer_index

    def forward(self, x):
        x = self.linear_layer(x)

        # The batch norm layer is applied only when the batch size is greater than 1
        if x.shape[0] > 1:
            x = self.batch_norm(x)

        if self.activation_exists:
            x = self.activation(x)
        return x

# mat = torch.tensor([[0, 1, 2, 3]], dtype=torch.float32)   
# model = IntermediateEncoder(4, 10, linear_layer={"bias": True}, batch_norm_layer={"momentum": 0.1}, activation_layer={"activation": nn.ReLU, "activation_args": {}})
# model(mat)
