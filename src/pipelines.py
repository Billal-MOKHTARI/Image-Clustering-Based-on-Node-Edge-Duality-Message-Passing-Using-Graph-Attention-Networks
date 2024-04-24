from torch_model_manager import TorchModelManager, NeptuneManager
import torch
from torch import nn
from typing import List
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import tempfile
import shutil
from torchvision import models

def visualize_models_hidden_layers(models : List[nn.Module], 
                                   image_path : str, 
                                   run, 
                                   neptune_manager : NeptuneManager, 
                                   names: List[str], 
                                   path : str = 'images', 
                                   instance_indexes: List[nn.Module] = [nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d],
                                   torch_transforms = None):
    
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
    >>> nm = NeptuneManager(project = "Billal-MOKHTARI/Image-Clustering-based-on-Dual-Message-Passing",
    ...                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NGRlOTNiZC0zNGZlLTRjNWUtYWEyMC00NzEwOWJkOTRhODgifQ==",
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

    image = Image.open(image_path)

    transformations = [transforms.ToTensor()]
    if torch_transforms is not None:
        transformations.extend(torch_transforms)

    transform = transforms.Compose(transformations)
    # Apply transformations
    image = transform(image)

    for model, name in zip(models, names):
        
        tmm = TorchModelManager(model)
        indexes = tmm.get_layer_by_instance(instance_indexes).keys()
        tmm.show_hidden_layers(torch.stack([image]), 
                               indexes = indexes, 
                               show_figure=False, 
                               run=run, 
                               neptune_manager=neptune_manager, 
                               image_workspace=f'{path}/{name}')
        

def create_embeddings(model, data_path, run, neptune_workspace, embedding_file_name, **kwargs):
    """
    Applies the model on the data and uploads the embeddings to Neptune.

    Args:
        model (nn.Module): PyTorch model.
        data_path (str): Path to the data.
        run: Neptune run object.
        neptune_workspace (str): Folder in Neptune where the embeddings should be uploaded.
    """
    torch_transforms = kwargs.get('torch_transforms', None)
    batch_size = kwargs.get('batch_size', 64)

    # Define transformations
    transformations = [transforms.ToTensor()]
    if torch_transforms is not None:
        transformations.extend(torch_transforms)

    # Load the dataset
    dataset = datasets.ImageFolder(data_path, transform=transformations)

    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    # Switch model to evaluation mode
    model.eval()

    # Iterate over data
    with torch.no_grad():
        for images, _ in data_loader:
            # Forward pass
            outputs = model(images)
            embeddings.append(outputs)

    # Concatenate embeddings
    embeddings = torch.cat(embeddings)
    # Save embeddings to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        torch.save(embeddings, embedding_file_name)


    # Track embeddings file in Neptune
    run[neptune_workspace].track_files(embedding_file_name)

    # Delete the temporary file
    os.remove(embedding_file_name)

model = models.vgg16(pretrained=True)
data_path = "../benchmark/agadez"

nm = NeptuneManager(project = "Billal-MOKHTARI/Image-Clustering-based-on-Dual-Message-Passing",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NGRlOTNiZC0zNGZlLTRjNWUtYWEyMC00NzEwOWJkOTRhODgifQ==",
                    run_ids_path="../configs/run_ids.json")
run = nm.create_run("data_visualization")

neptune_workspace = "embeddings"

create_embeddings(model, data_path, run, neptune_workspace, embedding_file_name, **kwargs)