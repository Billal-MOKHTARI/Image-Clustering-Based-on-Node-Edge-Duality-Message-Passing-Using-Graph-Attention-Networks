import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pipelines import viz_hidden_layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env import image_gat_mp_run
from torchvision import models
from torchvision import transforms

models = [models.vgg16(pretrained=True),
          models.vgg19(pretrained=True),
          models.resnet18(pretrained=True),
          models.efficientnet_b7(pretrained=True),
          models.convnext_large(pretrained=True),
          models.mobilenet_v3_large(pretrained=True),
          models.mnasnet1_3(pretrained=True)]

image_path = "/home/billalmokhtari/Documents/projects/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/benchmark/datasets/agadez/images/G0041951.JPG"
viz_hidden_layers(models = models, 
                  image_path=image_path, 
                  run=image_gat_mp_run, 
                  namespaces=["images/vgg16", 
                              "images/vgg19", 
                              "images/resnet18", 
                              "images/efficientnet_b7", 
                              "images/convnext_large", 
                              "images/mobilenet_v3_large", 
                              "images/mnasnet1_3"], 
                  
                  torch_transforms=[transforms.Resize((512, 512)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])