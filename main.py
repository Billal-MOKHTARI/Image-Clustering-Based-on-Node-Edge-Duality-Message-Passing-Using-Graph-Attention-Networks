from env import neptune_manager
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision import models,transforms
from PIL import Image
import seaborn as sns

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

plt.show()