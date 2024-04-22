import pickle
import json
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

def read_images(folder_path, n):
    """
    Read the first n images from a folder, resize them to (3, 224, 224), and stack them into a single tensor.

    Args:
        folder_path (str): Path to the folder containing images.
        n (int): Number of images to read.

    Returns:
        torch.Tensor: Stacked tensor of shape (N, 3, 224, 224).
    """
    images = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    for filename in sorted(os.listdir(folder_path)):
        if len(images) == n:
            break
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                img_tensor = transform(img)
                images.append(img_tensor)
    stacked_tensor = torch.stack(images, dim=0)
    return stacked_tensor


def dump_data(data, file_path: str) -> None:
    """
    Serialize and dump the data to a binary file using pickle.

    Parameters:
    - data: Any Python object to be serialized and saved.
    - file_path (str): The path to the file where data will be saved.
    """
    assert isinstance(file_path, str), "file_path must be a string"
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    except (IOError, FileNotFoundError, PermissionError) as e:
        print(f"Error during dump_data: {e}")

def load_data_from_path(file_path: str):
    """
    Load and deserialize data from a binary file using pickle.

    Parameters:
    - file_path (str): The path to the file from which data will be loaded.

    Returns:
    - loaded_data: The deserialized data.
    """
    assert isinstance(file_path, str), "file_path must be a string"

    try:
        # Load the data from the file
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
    

        return loaded_data

    except (IOError, FileNotFoundError, PermissionError, pickle.PickleError) as e:
        print(f"Error during load_data_from_path: {e}")
        return None
    

def parse_encoder(json_file_path, network_type):
    if network_type == "encoder":
        conv_prefix = ""
        pool_prefix = ""
    elif network_type == "decoder":
        conv_prefix = "de"
        pool_prefix = "un"
    
    # Open the JSON file
    with open(json_file_path, "r") as json_file:
        # Load the JSON data
        data = json.load(json_file)
        
    data["activation"] = getattr(nn, data["activation"])
    for i in range(len(data["shapes"])):
        data["shapes"][i] = tuple(data["shapes"][i])
    
    try:
        attr = conv_prefix+"conv_kernels"
        data[attr][0][0]
        for i, conv_kernel in enumerate(data[attr]):
            data[attr][i] = tuple(conv_kernel)
    except:
        data[attr] = tuple(data[attr])
    
    try:
        attr = conv_prefix+"conv_strides"
        data[attr][0][0]
        for i, conv_stride in enumerate(data[attr]):
            data[attr][i] = tuple(conv_stride)
    except:
        data[attr] = tuple(data[attr])
        
    
    try:
        attr = pool_prefix+"pool_strides"
        data[attr][0][0]
        for i, pool_stride in enumerate(data[attr]):
            data[attr][i] = tuple(pool_stride)
    except:
        data[attr] = tuple(data[attr])
        
    
    try:
        attr = pool_prefix+"pool_paddings"
        data[attr][0][0]
        for i, pool_padding in enumerate(data[attr]):
            data[attr][i] = tuple(pool_padding)
    except:
        data[attr] = tuple(data[attr])
        
    if network_type == "decoder":
        try:
            attr = "deconv_paddings"
            data[attr][0][0]
            for i, deconv_padding in enumerate(data[attr]):
                data[attr][i] = tuple(deconv_padding)
        except:
            data[attr] = tuple(data[attr])
    
    
    return data