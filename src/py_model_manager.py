import torchvision.models as models
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class PyModelManager:
    
    def __init__(self, model):
        self.model = model
        self.named_layers = dict()
        self.list_layers = list()
    
    def get_named_layers(self):
        def get_layers_recursive(model_children: dict):
            for name, layer in model_children.items():
                if dict(layer.named_children()) != {}:
                    model_children[name] = get_layers_recursive(dict(layer.named_children()))
                else:
                    model_children[name] = layer
            self.named_layers = model_children
            return self.named_layers
        return get_layers_recursive(dict(self.model.named_children()))
    

    def get_list_layers(self):

        self.list_layers = list(self.model.children())
        return self.list_layers

    def get_layer_by_name(self, index):
        layers = self.get_named_layers()

        try:
            for ind in index:
                layer = layers[ind]
                layers = layer
            return layer
        except TypeError:
            print(f'The index {index} is out of range.')    
        

    def get_layer_by_index(self, block_index, layer_index=None):
        if layer_index is None:
            return self.get_list_layers()[block_index]
        else:
            return self.get_list_layers()[block_index][layer_index]
    
   
    def delete_layer_by_name(self, index):
        layer = self.get_layer_by_name(index)
        del layer
        
    
    def delete_layer_by_index(self, block_index, layer_index=None):
        if layer_index is None:
            del self.list_layers[block_index]
        else:
            del self.list_layers[block_index][layer_index]        
    
    def get_output_size(self, layer):
        assert hasattr(layer, 'out_features'), f'{layer} does not have the attribute out_features'
        return layer.out_features
    
    def get_input_size(self, layer):
        assert hasattr(layer, 'in_features'), f'{layer} does not have the attribute in_features'
        return layer.in_features
    
    
    
vgg16 = models.vgg16(pretrained=True)

def model_to_dict(model_children: dict):
    for name, layer in model_children.items():
        if dict(layer.named_children()) != {}:
            model_children[name] = model_to_dict(dict(layer.named_children()))
        else:
            model_children[name] = layer
    return model_children
        



# print(dict_layers)
model = PyModelManager(vgg16)
# model.delete_layer_by_name('classifier', -1)
model.delete_layer_by_name(['classifier'])
print(model.get_named_layers())