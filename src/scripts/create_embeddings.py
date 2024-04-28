import os
import sys
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.data_loaders.data_loader import create_combined_images
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pipelines import create_embeddings
from torchvision import models
from env import image_gat_mp_run

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

del vgg16.classifier[-1]
del vgg19.classifier[-1]

del vgg16.classifier[-1]
del vgg19.classifier[-1]

del vgg16.classifier[-1]
del vgg19.classifier[-1]


models = [vgg16, vgg19]
names = ["vgg16", "vgg19"]
namespaces = ["embeddings/vgg16", "embeddings/vgg19"]
batch_size = 16

create_embeddings(models=models,
                  run=image_gat_mp_run,
                  namespaces=namespaces,
                  data_path="../../benchmark/datasets/agadez/images",
                  row_index_namespace="embeddings/row_index",
                  torch_transforms=[transforms.Resize((512, 512)), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])],
                  batch_size=batch_size)