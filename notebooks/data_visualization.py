import torch
from torch import nn
from torchvision import models, transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp, LayerCAM

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.data_loaders import data_loader

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
print(model)
# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),            # Normalize images using mean and standard deviation
])

dataset = data_loader.ImageFolderNoLabel('../benchmark/datasets/agadez', transform=transform)

# Create a data loader to load images in batches
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the data loader to access the images
for images in data_loader:
    # Do something with the images (e.g., pass them through a neural network)
    print(images.shape)


cam_extractor = SmoothGradCAMpp(model, 'features')
layer_extractor = LayerCAM(model, [model.features[0], model.features[1]])

out = model(images[0].unsqueeze(0))



cams = layer_extractor(out.squeeze(0).argmax().item(), out)
print(cams[1])
for name, cam in zip(layer_extractor.target_names, cams):
    plt.imshow(cam.squeeze(0).numpy()); plt.axis('off'); plt.title(name); plt.show()
