import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import numpy as np
import math
from tqdm import tqdm
import torch

def required_kernel(in_size: int, out_size:int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert out_size > 0, "Output size must be greater than 0"
    assert in_size >= out_size, "Input size must be greater than or equal to output size"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (1-out_size)*stride+in_size+2*padding

def required_kernel_transpose(in_size: int, out_size:int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert out_size > 0, "Output size must be greater than 0"
    assert in_size <= out_size, "Input size must be less than or equal to output size"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return out_size-(in_size-1)*stride+2*padding

def get_output(in_size: int, kernel_size: int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert kernel_size > 0, "Kernel size must be greater than 0"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (in_size-kernel_size+2*padding)//stride+1

def pe(i, j, M, N, mask, factor=100):
    phi = 1+(pow(-1, i)+pow(-1, j))/2
    psi = 2-phi
    ind_i = M*i+j
    ind_j = N*j+i
    theta = 2*math.pi*(i/pow(M, 2*ind_i/(factor*M*N))+j/pow(N, 2*ind_i/(factor*M*N)))+(i+j)/(M+N)**2


    pe = (mask/4)*(phi*math.cos(theta)+psi*math.sin(theta))**2
    return pe

def image_pe(matrix):
    if isinstance(matrix, torch.Tensor):
        lib = "torch"
        positional_matrix = torch.zeros_like(matrix)
    else:
        lib = "numpy"
        positional_matrix = np.zeros_like(matrix)
    
    M, N = matrix.shape

    for i in tqdm(range(M), desc="rows", unit="row"):
        for j in range(N):
            positional_matrix[i, j] = pe(i, j, M, N, matrix[i, j], lib)
    
    if lib == "torch":
        positional_matrix /= torch.max(positional_matrix)
    else:
        positional_matrix /= np.max(positional_matrix)
    
    return positional_matrix