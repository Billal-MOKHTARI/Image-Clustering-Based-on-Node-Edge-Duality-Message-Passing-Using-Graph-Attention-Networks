from typing import List, Union
import pandas as pd
import torch
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models import constants

def extract_channels(tensor, index=None, columns=None) -> List[pd.DataFrame]:
    channels = []
    shape = tensor.shape
    assert len(shape) == 3, "The matrix should be a 3D tensor"

    for i in range(shape[0]):
        channel = tensor[i, :, :]
        channel = pd.DataFrame(channel, index=index, columns=columns)
        channels.append(channel)

    return channels


def required_kernel(in_size: int, out_size:int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert out_size > 0, "Output size must be greater than 0"
    assert in_size >= out_size, "Input size must be greater than or equal to output size"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (1-out_size)*stride+in_size+2*padding

def get_output_model(model, input_shape):
    # Assuming 'model' is your model and 'input_shape' is the shape of your input
    dummy_input = torch.randn(1, *input_shape)  # Create a dummy input
    output = model(dummy_input)  # Pass the dummy input through the model
    output_size = output.size()  # Get the size of the output
    if output_size[0] == 1 and len(output_size) == 2:
        output_size = output_size[1]
    
    return output_size

def intersect_list(list1, list2):
    intersection = []

    for i in list1:
        if i in list2:
            intersection.append(i)
    
    return intersection

def create_dual_adjacency_tensor(primal_adjacency_tensor, primal_index, delimiter):
        
        def sort_and_fill_matrix(matrix, large_index):
            # Sort the index of the matrix according to the large index
            sorted_index = sorted(matrix.index, key=lambda x: large_index.index(x))
            sorted_columns = sorted(matrix.columns, key=lambda x: large_index.index(x))
            
            # Reindex the matrix
            matrix = matrix.reindex(index=sorted_index, columns=sorted_columns)
            
            # Add empty rows and columns where indexes of the matrix are missing
            missing_rows = [index for index in large_index if index not in matrix.index]
            missing_columns = [index for index in large_index if index not in matrix.columns]
            
            for row in missing_rows:
                matrix.loc[row] = [0] * len(matrix.columns)
                
            for col in missing_columns:
                matrix[col] = [0] * len(matrix.index)
            
            # Reorder rows and columns to match large_index
            matrix = matrix.reindex(index=large_index, columns=large_index)
            
            return matrix
        def create_dual(matrix, delimiter):
        
            dual_index = create_index(matrix, delimiter)
            size = len(dual_index)

            init_data = np.zeros((size, size))
            dual_matrix = pd.DataFrame(init_data, index=dual_index, columns=dual_index)

            for ind1 in dual_index:
                for ind2 in dual_index:
                    ind1_split = ind1.split(delimiter)
                    ind2_split = ind2.split(delimiter)

                    intersection = intersect_list(ind1_split, ind2_split)

                    if len(intersection) == 2:
                        dual_matrix.loc[ind1, ind2] = 1
                    
                    elif intersection != []:
                        dual_matrix.loc[ind1, ind2] = 1
                        
            return dual_matrix
        def create_index(matrix, delimiter):
            assert matrix.shape[0] == matrix.shape[1], "The matrix should be square"
            rank = matrix.shape[0]
            columns = matrix.columns
            
            new_index = []

            assert [delimiter not in column for column in columns], "The delimiter should not be present in any column"

            np_matrix = matrix.to_numpy()

            is_symmetric = np.allclose(np_matrix, np_matrix.T)
            if not is_symmetric:
                k = 0

            for i in range(rank):
                if is_symmetric:
                    k = i
                for j in range(k+1, rank):
                    if np_matrix[i][j] != 0:
                        new_index.append(f"{columns[i]}{delimiter}{columns[j]}")

            return new_index
        
        def create_dual_index(primal_adjacency_tensor, primal_index, delimiter):
            assert len(primal_adjacency_tensor.shape) == 3, "The tensor should be a 3D tensor"
            assert primal_adjacency_tensor.shape[1] == primal_adjacency_tensor.shape[2], "The tensor should be square"

            channels = extract_channels(primal_adjacency_tensor.detach().numpy(), primal_index, primal_index)

            new_index = list()
            for channel in channels:
                new_index = list(set(new_index).union(set(create_index(channel, delimiter))))

            return sorted(new_index)

        dual_index = create_dual_index(primal_adjacency_tensor, primal_index, delimiter)
        channels = extract_channels(primal_adjacency_tensor, primal_index, primal_index)

        dual_channels = []

        ind = 0
        for channel in channels:
            
            dual_channel = create_dual(channel, delimiter)
            dual_channel = sort_and_fill_matrix(dual_channel, dual_index)
            dual_channels.append(dual_channel)
            ind += 1
        
        dual_adjacency_tensor = torch.tensor(np.array(dual_channels), dtype=constants.FLOATING_POINT)
        dual_nodes = []
        for ind in dual_index:
            row, col = ind.split(delimiter)
            row, col = primal_index.index(row), primal_index.index(col)
            dual_nodes.append(primal_adjacency_tensor[:, row, col])
        
        dual_nodes = torch.stack(dual_nodes, dim=0)
        dual_nodes.to(dtype=constants.FLOATING_POINT)
        
        return dual_index, dual_adjacency_tensor, dual_nodes

def list_sum(l):
    return sum(list(filter(lambda x: (x),l)))



def load_data(path):
    data = pd.read_csv(path)
    return data

def create_index(matrix, delimiter):
    assert matrix.shape[0] == matrix.shape[1], "The matrix should be square"
    rank = matrix.shape[0]
    columns = matrix.columns
    new_index = []

    assert [delimiter not in column for column in columns], "The delimiter should not be present in any column"

    np_matrix = matrix.to_numpy()

    is_symmetric = np.allclose(np_matrix, np_matrix.T)
    if not is_symmetric:
        k = 0

    for i in range(rank):
        if is_symmetric:
            k = i
        for j in range(k+1, rank):
            if np_matrix[i][j] != 0:
                new_index.append(f"{columns[i]}{delimiter}{columns[j]}")

    return new_index

def intersect_list(list1, list2):
    intersection = []

    for i in list1:
        if i in list2:
            intersection.append(i)
    
    return intersection
            


def create_dual(matrix, delimiter, mapper):
    # assert all(isinstance(i, (float, int)) for i in mapper.values()), "All elements must be floats"
    dual_index = create_index(matrix, delimiter)
    size = len(dual_index)

    init_data = np.zeros((size, size))
    dual_matrix = pd.DataFrame(init_data, index=dual_index, columns=dual_index)

    for ind1 in dual_index:
        for ind2 in dual_index:
            ind1_split = ind1.split(delimiter)
            ind2_split = ind2.split(delimiter)

            intersection = intersect_list(ind1_split, ind2_split)

            if len(intersection) == 2:
                dual_matrix.loc[ind1, ind2] = 1
            
            elif intersection != []:
                dual_matrix.loc[ind1, ind2] = mapper.loc[intersection[0]]
                
    return dual_matrix



def construct_dual_index(tensor, index, delimiter):
    assert len(tensor.shape) == 3, "The tensor should be a 3D tensor"
    assert tensor.shape[1] == tensor.shape[2], "The tensor should be square"

    channels = extract_channels(tensor, index, index)

    new_index = list()
    for channel in channels:
        new_index = list(set(new_index).union(set(create_index(channel, delimiter))))
    
    return new_index

def sort_and_fill_matrix(matrix, large_index):
    # Sort the index of the matrix according to the large index
    sorted_index = sorted(matrix.index, key=lambda x: large_index.index(x))
    sorted_columns = sorted(matrix.columns, key=lambda x: large_index.index(x))
    
    # Reindex the matrix
    matrix = matrix.reindex(index=sorted_index, columns=sorted_columns)
    
    # Add empty rows and columns where indexes of the matrix are missing
    missing_rows = [index for index in large_index if index not in matrix.index]
    missing_columns = [index for index in large_index if index not in matrix.columns]
    
    for row in missing_rows:
        matrix.loc[row] = [0] * len(matrix.columns)
        
    for col in missing_columns:
        matrix[col] = [0] * len(matrix.index)
    
    # Reorder rows and columns to match large_index
    matrix = matrix.reindex(index=large_index, columns=large_index)
    
    return matrix

def convert_list(list_of_objects, source_type, destination_type):
    """
    Convert a list of objects from source_type to destination_type.

    Args:
        list_of_objects (list): List of objects to be converted.
        source_type (type): Source type of the objects.
        destination_type (type): Destination type to which objects will be converted.

    Returns:
        list: List of objects converted to the destination_type.
    """
    # Convert objects using list comprehension
    return [destination_type(obj) for obj in list_of_objects]

def create_dual_tensor(tensor, index, delimiter, mapper):
    assert mapper.shape[0] == len(index), "The number of rows in the mapper should be equal to the number of indexes"
    
    dual_index = construct_dual_index(tensor, index, delimiter)

    channels = extract_channels(tensor, index, index)
    assert mapper.shape[1] == len(channels), "The number of columns in the mapper should be equal to the number of channels"
    dual_channels = []

    ind = 0
    for channel in channels:
        channel_mapper = mapper.iloc[:, ind]
        dual_channel = create_dual(channel, delimiter, channel_mapper)
        dual_channel = sort_and_fill_matrix(dual_channel, dual_index)

        dual_channels.append(dual_channel)
        ind += 1
    
    return dual_index, torch.tensor(np.array(dual_channels))
        