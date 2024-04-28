from torch_model_manager import TorchModelManager

import torch
from torch import nn
from typing import List
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pickle
from models.data_loaders import data_loader as dl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env import neptune_manager

def viz_hidden_layers(models : List[nn.Module], 
                    image_path : str, 
                    run: str, 
                    namespaces: List[str], 
                    instance_indexes: List[nn.Module] = [nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d],
                    torch_transforms = None,
                    method = "layercam",
                    models_from_path = False):
    
    """
    Visualizes the hidden layers of multiple PyTorch models.

    Args:
        models (List[nn.Module]): List of PyTorch models.
        image_path (str): Path to the input image.
        run: (Type of 'run' variable, such as neptune.run): Neptune run object.
        neptune_manager (NeptuneManager): Neptune manager object.
        names (List[str]): Names of the models for identification.
        path (str, optional): Path to save the visualizations. Defaults to 'images'.
        instance_indexes (List[nn.Module], optional): List of PyTorch module instances for indexing. Defaults to [nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d].
        torch_transforms: (Optional): Optional list of Torch transformations to apply to the image before passing through the models.

    Example:
    >>> image_path = "PHOTO-2023-10-24-16-06-49-1.jpg"

    >>> # Create a project
    >>> nm = NeptuneManager(project = "...",
    ...                     api_token="...",
    ...                     run_ids_path="../configs/run_ids.json")
    >>> run = nm.create_run("data_visualization")

    >>> # Visualize hidden layers of each backbone
    >>> vgg16 = models.vgg16(pretrained=True)
    >>> vgg19 = models.vgg19(pretrained=True)
    >>> resnet18 = models.resnet18(pretrained=True)
    >>> efficientnet_b7 = models.efficientnet_b7(pretrained=True)
    >>> convnext_large = models.convnext_large(pretrained=True)
    >>> mobile_net_v3_large = models.mobilenet_v3_large(pretrained=True)
    >>> maxvit_t = models.maxvit_t(pretrained=True)
    >>> vit_l_32 = models.vit_l_32(pretrained=True)
    >>> mnasnet_1_3 = models.mnasnet1_3(pretrained=True)
    >>> models = [efficientnet_b7]

    >>> visualize_models_hidden_layers(models=models, 
    ...                                image_path=image_path, 
    ...                                run=run, 
    ...                                neptune_manager=nm, 
    ...                                path='images',
    ...                                names=["efficientnet_b7"])
    """
    run = neptune_manager.Run(run)
    
    # Load the image from the path
    image = Image.open(image_path)

    # Define transformations and apply them on the image
    transformations = [transforms.ToTensor()]
    if torch_transforms is not None:
        transformations.extend(torch_transforms)

    transform = transforms.Compose(transformations)
    image = transform(image)

    # Log the hidden layers and save the visualizations if save_paths is not None

    for model, namespace in tqdm(zip(models, namespaces), total=len(models)):
        
        tmm = TorchModelManager(model)
        indexes = tmm.get_layer_by_instance(instance_indexes).keys()
        run.log_hidden_conv2d(model = model,
                            input_data=torch.stack([image]), 
                            indexes = indexes, 
                            method=method,
                            namespace=namespace)
        

def create_embeddings(models, run, namespaces, data_path, row_index_namespace=None, **kwargs):
    """
    Applies the model on the data and uploads the embeddings to Neptune.

    Args:
        model (nn.Module): PyTorch model.
        neptune_manager (NeptuneManager): Neptune manager object.
        run: Neptune run object.
        namespace (str): Folder in Neptune where the embeddings should be uploaded.
        data_path (str): Path to the data.
        embedding_file_name (str): Name of the file to save the embeddings.
        **kwargs: Additional keyword arguments.
            torch_transforms (list, optional): List of Torch transformations to apply to the image before passing through the models. Defaults to None.
            batch_size (int, optional): Batch size for data loader. Defaults to 64.
            keep (bool, optional): Whether to keep the temporary file after tracking it in Neptune. Defaults to False.

    """
    run = neptune_manager.Run(run)
    torch_transforms = kwargs.get('torch_transforms', None)
    batch_size = kwargs.get('batch_size', 64)
    
    # Define transformations
    transformations = [transforms.ToTensor()]
    if torch_transforms is not None:
        transformations.extend(torch_transforms)

    # Create a data loader
    data_loader = dl.ImageFolderNoLabel(data_path, transform=transforms.Compose(transformations))
    row_index = data_loader.get_paths

    for model, namespace in tqdm(zip(models, namespaces), desc="Models"):
        data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)
        embeddings = []
        
        # Switch model to evaluation mode
        model.eval()

        # Iterate over data
        for images in tqdm(data_loader, desc="Data"):
            # Forward pass
            with torch.no_grad():
                outputs = model(images)
                embeddings.append(outputs)

        # Concatenate embeddings
        embeddings = torch.cat(embeddings)
        # Save embeddings to a temporary file
        run.log_files(data=embeddings, namespace=namespace)
        if row_index_namespace is not None:
            run.log_files(data=row_index, namespace=row_index_namespace)


# def prepare_data(annotation_path, 
#                  image_path, 
#                  output_path, 
#                  model, 
#                  namespace, 
#                  neptune_manager = NEPTUNE_MANAGER, 
#                  run = DATA_VISUALIZATION_RUN, 
#                  transpose=True, 
#                  **kwargs):
    
#     adjacency_tensor, index_row, index_col = dl.annotation_matrix_to_adjacency_tensor(from_csv=annotation_path, transpose=transpose)

#     try:
#         neptune_manager.delete_data(run, [namespace])
#     except:
#         pass

#     embeddings = create_embeddings(model=model, 
#                                    neptune_manager=neptune_manager, 
#                                    run=run, 
#                                    namespace=namespace, 
#                                    data_path=image_path,
#                                    output_path=output_path,
#                                    **kwargs)
    
#     result = {
#         "adjacency_tensor": adjacency_tensor,
#         "index_row": index_row,
#         "index_col": index_col,
#         "embeddings": embeddings
#     }

#     return result