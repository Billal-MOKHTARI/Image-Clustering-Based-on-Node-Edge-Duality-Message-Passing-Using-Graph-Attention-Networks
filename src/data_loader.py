import pandas as pd
import numpy as np

def load_data(path):
    data = pd.read_csv(path)
    return data

def create_index(matrix, delimiter, scalar=True):
    assert matrix.shape[0] == matrix.shape[1], "The matrix should be square"
    rank = matrix.shape[0]
    columns = matrix.columns
    new_index = []

    assert [delimiter not in column for column in columns], "The delimiter should not be present in any column"

    np_matrix = matrix.to_numpy()
    null_value = 0 if scalar else []

    is_symmetric = np.allclose(np_matrix, np_matrix.T)
    if not is_symmetric:
        k = 0

    for i in range(rank):
        if is_symmetric:
            k = i
        for j in range(k, rank):
            if np_matrix[i][j] != null_value:
                new_index.append(f"{columns[i]}{delimiter}{columns[j]}")

    return new_index

def create_dual(matrix, delimiter, scalar=True):
    assert matrix.shape[0] == matrix.shape[1], "The matrix should be square"
    rank = matrix.shape[0]
    columns = matrix.columns
    new_index = []

    assert [delimiter not in column for column in columns], "The delimiter should not be present in any column"

    

# A = pd.DataFrame([[1, 0, 2, 3], 
#                   [0, 4, 5, 6], 
#                   [2, 5, 7, 8], 
#                   [3, 6, 8, 9]], columns=["A", "B", "C", "D"], index=["A", "B", "C", "D"])
# print(A)
# print(create_index(A, "_", scalar=True))
