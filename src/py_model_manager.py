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

    def get_list_layers(self, data):    
        pass

    def get_layer(self, indexes):
        trace = ['self.model']
        for index in indexes[:-1]:
            
            if isinstance(index, int):
                trace.append(f'[{index}]')
            else:
                trace.append(f'.{index}')
        
        if isinstance(indexes[-1], int):
            layers = eval(''.join(trace))
            return layers[indexes[-1]]
        else:
            return getattr(eval(''.join(trace)), indexes[-1])

    def delete_layer(self, indexes):
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

    def get_layer_by_index(self, block_index, layer_index=None):
        if layer_index is None:
            return self.get_list_layers()[block_index]
        else:
            return self.get_list_layers()[block_index][layer_index]
    

            
    
    # Get properties of the layer
    def get_output_size(self, layer):
        assert hasattr(layer, 'out_features'), f'{layer} does not have the attribute out_features'
        return layer.out_features
    
    def get_input_size(self, layer):
        assert hasattr(layer, 'in_features'), f'{layer} does not have the attribute in_features'
        return layer.in_features
    def get_attribute(self, layer, attribute):
        assert hasattr(layer, attribute), f'{layer} does not have the attribute {attribute}'
        return getattr(layer, attribute)
    
    # Search for a layer in the model by property
    def search_layer(self, property, value):
        def dfs(self, model, property, value, layers, indexes, tmp=[]):
            for name, layer in model.named_children():
                print(dict(layer.named_children()))

                if hasattr(layer, property) and self.get_attribute(layer, property) == value:
                    layers.append(layer)
                    indexes.append(tmp)
                    tmp = []
                else:
                    tmp.append(name)
                    dfs(self, layer, property, value, layers, indexes, tmp)

        layers = []
        indexes = []

        dfs(self, self.model, property, value, layers, indexes)
        return layers, indexes
    
    def delete_by_attribute(self, property, value):
        pass
    
vgg = models.vgg16(pretrained=True)

model = PyModelManager(vgg)
# model.delete_layer_recursive(['classifier', -1])
print(model.search_layer('out_features', 4096))
# print(vgg.classifier[0])
