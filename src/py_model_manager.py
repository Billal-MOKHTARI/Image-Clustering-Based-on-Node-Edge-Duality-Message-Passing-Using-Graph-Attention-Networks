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

        self.dict_layers = dict(self.model.named_children())
        return self.dict_layers

    def get_list_layers(self):

        # Get the layers of the pre-trained model
        self.list_layers = list(self.model.children())
        return self.list_layers

    def get_layer_by_name(self, layer_name, layer_index=None):
        
        
        if layer_index is None:
            return self.get_named_layers()[layer_name]
        else:
            return self.get_named_layers()[layer_name][layer_index]

    def get_layer_by_index(self, block_index, layer_index=None):
        if layer_index is None:
            return self.get_list_layers()[block_index]
        else:
            return self.get_list_layers()[block_index][layer_index]
    
    def delete_layer_by_name(self, layer_name, layer_index=None):
        if layer_index is None:
            delattr(self.model, layer_name)
        else:
            layers = self.get_named_layers()[layer_name]
            del layers[layer_index]
    
    def delete_layer_recursive(self, indexes):
        trace = ['self.model']
        for index in indexes[:-1]:
            
            if isinstance(index, int):
                trace.append(f'[{index}]')
            else:
                trace.append(f'.{index}')
        
        if isinstance(indexes[-1], int):
            layers = eval(''.join(trace))
            del layers[indexes[-1]]
        else:
            delattr(eval(''.join(trace)), indexes[-1])
            
    
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
    
vgg = models.vgg16(pretrained=True)

model = PyModelManager(vgg)
model.delete_layer_recursive(['classifier', -1])
print(model.get_named_layers())
# print(vgg.classifier[0])
