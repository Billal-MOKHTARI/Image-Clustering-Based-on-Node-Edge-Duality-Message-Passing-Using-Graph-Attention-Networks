import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.data_loaders.data_loader import create_combined_images
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pipelines import create_embeddings
from torchvision import models
from env import image_gat_mp_run

models = [models.vgg16(pretrained=True), models.vgg19(pretrained=True)]
names = ["vgg16", "vgg19"]

create_embeddings(models=models,
                  run=image_gat_mp_run,
                  namespaces=["embeddings/vgg16", "embeddings/vgg19"],
                  data_path="/home/billalmokhtari/Documents/projects/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/benchmark/datasets/agadez/images",
                  row_index_namespace="embeddings/row_index")