import torch
from torchvision import models, transforms
import os
import matplotlib.pyplot as plt
from torch_model_manager import TorchModelManager, NeptuneManager
from PIL import Image
from torch import nn

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.data_loaders import data_loader
from src import utils



def show_images(images, num_rows, num_cols, titles=None, scale=1.5, grayscale=False):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if grayscale and img.size(0) == 1:
            print(img.shape)
            print(img.numpy().reshape(1, img.shape[0], img.shape[1]))
            ax.imshow(img.numpy().reshape(1, img.shape[0], img.shape[1]), cmap='gray')
        else:
            ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

model = models.vgg16(pretrained=True).eval()

# Load image
image_path = "PHOTO-2023-10-24-16-06-49-1.jpg"
image = Image.open(image_path)
# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
# Apply transformations
image = transform(image)

# Create a project
nm = NeptuneManager(project = "Billal-MOKHTARI/Image-Clustering-based-on-Dual-Message-Passing",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NGRlOTNiZC0zNGZlLTRjNWUtYWEyMC00NzEwOWJkOTRhODgifQ==",
                    run_ids_path="../configs/run_ids.json")
run = nm.create_run("data_visualization")

# Visualize hidden layers of each backbone
vgg16 = models.vgg16(pretrained=True)
vgg19 = models.vgg19(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
efficientnet_b7 = models.efficientnet_b7(pretrained=True)
convnext_large = models.convnext_large(pretrained=True)
mobile_net_v3_large = models.mobilenet_v3_large(pretrained=True)
maxvit_t = models.maxvit_t(pretrained=True)
vit_l_32 = models.vit_l_32(pretrained=True)
mnasnet_1_3 = models.mnasnet1_3(pretrained=True)

models = [efficientnet_b7]

def visualize_models_hidden_layers(models, image, run, neptune_manager, names, path = 'images', 
                             instance_indexes = [nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d]):
    

    for model, name in zip(models, names):
        
        tmm = TorchModelManager(model)
        indexes = tmm.get_layer_by_instance(instance_indexes).keys()
        tmm.show_hidden_layers(torch.stack([image]), 
                               indexes = indexes, 
                               show_figure=False, 
                               run=run, 
                               neptune_manager=neptune_manager, 
                               image_workspace=f'{path}/{name}')



visualize_models_hidden_layers(models=models, 
                               image=image, 
                               run=run, 
                               neptune_manager=nm, 
                               path = 'images',
                               names = ["efficientnet_b7"])

