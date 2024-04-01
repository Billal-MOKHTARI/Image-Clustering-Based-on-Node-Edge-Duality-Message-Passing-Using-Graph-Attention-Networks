import torchvision.models as models
from torch import nn
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from . import utils

class PyModelManager:
    """
    A class for managing PyTorch models.

    Attributes:
        model: The PyTorch model to manage.
        named_layers: A dictionary containing the named layers of the model.
    """

    def __init__(self, model):
        """
        Initialize PyModelManager with a given PyTorch model.

        Args:
            model: The PyTorch model to manage.
        """
        self.model = model
        self.named_layers = dict()
    
    def get_named_layers(self):
        """
        Recursively fetch and store all named layers of the model.

        Returns:
            dict: A dictionary containing the named layers of the model.
        """
        def get_layers_recursive(model_children: dict):
            for name, layer in model_children.items():
                if dict(layer.named_children()) != {}:
                    model_children[name] = get_layers_recursive(dict(layer.named_children()))
                else:
                    model_children[name] = layer
            self.named_layers = model_children
            return self.named_layers
        
        return get_layers_recursive(dict(self.model.named_children()))

    def get_layer_by_index(self, index):
        """
        Get a layer from the model using its index.

        Args:
            index (list): The index path of the layer in the model.

        Returns:
            torch.nn.Module: The layer from the model.
        """
        trace = ['self.model']
        try:
            for ind in index[:-1]:
                
                if isinstance(ind, int):
                    trace.append(f'[{ind}]')
                else:
                    trace.append(f'.{ind}')
            
            if isinstance(index[-1], int):
                layers = eval(''.join(trace))
                return layers[index[-1]]
            else:
                return getattr(eval(''.join(trace)), index[-1])
        except:
            print(f'Layer with index {index} not found')
            

    def delete_layer_by_index(self, indexes):
        """
        Delete a layer from the model using its index.

        Args:
            indexes (list): The index path of the layer in the model.
        """
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
    

    def get_attribute(self, layer, attribute):
        """
        Get a specific attribute of a layer.

        Args:
            layer (torch.nn.Module): The layer.
            attribute (str): The attribute name.

        Returns:
            Any: The value of the attribute.
        """
        assert hasattr(layer, attribute), f'{layer} does not have the attribute {attribute}'
        return getattr(layer, attribute)
    
    def search_layer_by_attribute(self, property, value):
        """
        Search for layers in the model with a specific attribute value.

        Args:
            property (str): The attribute to search for.
            value: The value of the attribute to match.

        Returns:
            dict: A dictionary mapping the indexes of found layers to the layers themselves.
        """
        def dfs(self, model, property, value, layers, indexes, tmp=None, depth=0):
            if tmp is None:
                tmp = []

            for name, layer in model.named_children():
                tmp.append(name)
                if hasattr(layer, property) and self.get_attribute(layer, property) == value:
                    layers.append(layer)
                    indexes.append(tmp.copy())

                # Recursive call
                dfs(self, layer, property, value, layers, indexes, tmp, depth+1)

                # Pop the last element to backtrack
                tmp.pop()

        layers = []
        indexes = []

        dfs(self, self.model, property, value, layers, indexes)
        
        return utils.create_dictionary(utils.convert_to_int(indexes), layers)
    
    def search_layer_by_instance(self, instance_type):
        """
        Search for layers in the model by their instance type.

        Args:
            instance_type (type): The instance type of the layers to search for.

        Returns:
            dict: A dictionary mapping the indexes of found layers to the layers themselves.
        """
        def dfs(self, model, instance_type, layers, indexes, tmp=None, depth=0):
            if tmp is None:
                tmp = []

            for name, layer in model.named_children():
                tmp.append(name)
                if isinstance(layer, instance_type):
                    layers.append(layer)
                    indexes.append(tmp.copy())

                # Recursive call
                dfs(self, layer, instance_type, layers, indexes, tmp, depth + 1)

                # Pop the last element to backtrack
                tmp.pop()

        layers = []
        indexes = []

        dfs(self, self.model, instance_type, layers, indexes)

        return utils.create_dictionary(utils.convert_to_int(indexes), layers)    

    def delete_layer_by_attribute(self, property, value):
        """
        Delete layers from the model with a specific attribute value.

        Args:
            property (str): The attribute to search for.
            value: The value of the attribute to match.
        """
        search_res = self.search_layer_by_attribute(property, value)
        for key in search_res.keys():
            self.delete_layer_by_index(key)

    def delete_layer_by_instance(self, instance_type):
        """
        Delete layers from the model by their instance type.

        Args:
            instance_type (type): The instance type of the layers to delete.
        """
        search_res = self.search_layer_by_instance(instance_type)
        for key in search_res.keys():
            self.delete_layer_by_index(key)
