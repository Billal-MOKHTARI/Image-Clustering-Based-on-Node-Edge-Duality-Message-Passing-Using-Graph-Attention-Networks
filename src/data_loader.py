import pandas as pd
import numpy as np
from typing import List
import utils

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

    channels = utils.extract_channels(tensor, index, index)

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

def create_dual_tensor(tensor, index, delimiter, mapper):
    assert mapper.shape[0] == len(index), "The number of rows in the mapper should be equal to the number of indexes"
    
    dual_index = construct_dual_index(tensor, index, delimiter)

    channels = utils.extract_channels(tensor, index, index)
    assert mapper.shape[1] == len(channels), "The number of columns in the mapper should be equal to the number of channels"
    dual_channels = []

    ind = 0
    for channel in channels:
        channel_mapper = mapper.iloc[:, ind]
        dual_channel = create_dual(channel, delimiter, channel_mapper)
        dual_channel = sort_and_fill_matrix(dual_channel, dual_index)

        dual_channels.append(dual_channel)
        ind += 1
    
    return dual_index, np.array(dual_channels)
        
    

index = ["1", "2", "3", "4", "5"]
vals = np.array([[8, 1, 5], [2, 22, 5], [0.2, 5, 7], [1, 4, 0], [1, 6, 0.9]])

mapper = pd.DataFrame(vals, index=index)

mat1 = np.array([[1, 1, 0, 1, 1], 
                [1, 1, 1, 0, 0], 
                [0, 1, 1, 0, 1], 
                [1, 0, 0, 1, 0],
                [1, 0, 1, 0, 1]])

mat2 = np.array([[1, 0, 1, 1, 1], 
                [0, 1, 1, 0, 1], 
                [1, 1, 1, 1, 1], 
                [1, 0, 1, 1, 0],
                [1, 1, 1, 0, 1]])

mat3 = np.array([[1, 1, 0, 0, 1], 
                [1, 1, 0, 1, 0], 
                [0, 0, 1, 0, 1], 
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1]])

mat = np.array([mat1, mat2, mat3])

dual_matrix = create_dual_tensor(mat, index, "_", mapper)
print(dual_matrix[0])
print(dual_matrix[1])