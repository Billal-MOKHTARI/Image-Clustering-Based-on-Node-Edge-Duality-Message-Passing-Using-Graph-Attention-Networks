from typing import List, Union
import pandas as pd
import torch

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
