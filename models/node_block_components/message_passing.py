from torch import nn
from torchsummary import summary
import torch
class MessagePassing(nn.Module):
    def __init__(self, graph_order, in_features, out_features, layer_index, **kwargs):
        super(MessagePassing, self).__init__()
        batch_norm_args_name = "batch_norm"
        activation_args_name = "activation"
        self.activation_exists = False

        self.layer_index = layer_index
        self.graph_order = graph_order


        if batch_norm_args_name in kwargs.keys():
            batch_norm_args = kwargs[batch_norm_args_name]
        if activation_args_name in kwargs.keys():
            activation_args = kwargs[activation_args_name]
            layer = activation_args["layer"]
            activation_args = activation_args["args"]

            self.activation_exists = True
            self.activation = layer(**activation_args)

        self.linear_layer = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features, **batch_norm_args)
        
        

    def get_layer_index(self):
        return self.layer_index

    def forward(self, x, adjacency_tensor):
        # H(k) * W_h(k)
        x = self.linear_layer(x)

        # The batch norm layer is applied only when the batch size is greater than 1
        if x.shape[0] > 1:
            x = self.batch_norm(x)

        

        if self.activation_exists:
            x = self.activation(x)
        return x

mat = torch.tensor([[0, 1, 2, 3]], dtype=torch.float32)   
model = MessagePassing(4, 10, linear_layer={"bias": True}, batch_norm={"momentum": 0.1}, activation={"layer": nn.ReLU, "args": {}}, layer_index=1)
print(model(mat))
