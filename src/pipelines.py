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
import pandas as pd
from torch_model_manager import SegmentationManager
from src import utils
from scipy.stats import hmean


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
    if run is not None:
        run = neptune_manager.Run(run)
    else:
        run = neptune_manager
    
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
        

def create_embeddings(models, namespaces, data_path, run = None, models_from_path = False, row_index_namespace=None, **kwargs):
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
    if run is not None:
        run = neptune_manager.Run(run)
    else:
        run = neptune_manager  

    torch_transforms = kwargs.get('torch_transforms', None)
    batch_size = kwargs.get('batch_size', 64)
    preprocess = kwargs.get('preprocess', [None for _ in models])
    
    # Define transformations
    transformations = [transforms.ToTensor()]
    if torch_transforms is not None:
        transformations.extend(torch_transforms)


    for model, namespace, prss in tqdm(zip(models, namespaces, preprocess), desc="Models"):

        # Create a data loader
        data_loader = dl.ImageFolderNoLabel(data_path, 
                                            transform=transforms.Compose(transformations) if prss== False else None,
                                            preprocess=prss)
        row_index = data_loader.get_paths()
        data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)
        embeddings = []
        
        # Switch model to evaluation mode
        if prss is None:
            model.eval()
 
        # Iterate over data
        for images in tqdm(data_loader, desc="Data"):
            # Forward pass
            
            if prss is None:
                outputs = model(images)
            else:
                outputs = model.encode_image(images)
            
            embeddings.append(outputs)
        # Concatenate embeddings
        embeddings = torch.cat(embeddings)
        # Save embeddings to a temporary file
        run.log_files(data=embeddings, namespace=namespace)
        if row_index_namespace is not None:
            run.log_files(data=row_index, namespace=row_index_namespace)

def create_adjacency_tensor(run, 
                            row_index_path: str, 
                            dataset_path, 
                            classes, 
                            box_threshold, 
                            text_threshold, 
                            nms_threshold, 
                            output_namespace,
                            agg=hmean, 
                            annotation_matrix_processing=utils.sort_dataframe):
    """
    Creates an adjacency tensor from an annotation matrix.

    Args:
        annotation_matrix_path (str): Path to the annotation matrix.

    Returns:
        torch.Tensor: Adjacency tensor.
    """
    if run is not None:
        run = neptune_manager.Run(run)
    else:
        run = neptune_manager 

    row_index = run.fetch_pkl_data(row_index_path)

    # annotation_matrix = utils.sort_dataframe(annotation_matrix, mode="rows", index=row_index)

    seg_manager = SegmentationManager()
    adjacency_tensor = seg_manager.occ_proba_disjoint_tensor(dataset_path=dataset_path, 
                                                             classes=classes,
                                                             box_threshold=box_threshold,
                                                             text_threshold=text_threshold,
                                                             nms_threshold=nms_threshold,
                                                             agg=agg,
                                                             annotation_matrix_processing=annotation_matrix_processing,
                                                             annotation_matrix_processing_args={"mode": "rows", "index": row_index})
    
    run.log_files(data=adjacency_tensor, namespace=output_namespace, extension='pkl', wait=True)
    return adjacency_tensor

