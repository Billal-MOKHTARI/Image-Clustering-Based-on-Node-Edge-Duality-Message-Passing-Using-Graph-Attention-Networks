import os
import sys
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from tqdm import tqdm
from pipelines import create_embeddings
from torchvision import models
import json
from typing import Union, List
import argparse
# Define a function that evaluate a list of string
def exec_code(code: Union[List[str], str]):
    if isinstance(code, str):
        code = code.split("\n")
    
    for line in code:
        exec(line)

parser = argparse.ArgumentParser(prog='create_embeddings', description='Create embeddings from a list of PyTorch models')
parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)

args = parser.parse_args()
path = args.config

with open(path, "r") as f:
    config = json.load(f)

pretrained_model_weights_parser_path = "../../configs/pretrained_model_weights_parser.json"

with open(pretrained_model_weights_parser_path) as f:
    model_weights_parser = json.load(f)

torch_models = config["models"]
models_from_path = config["models_from_path"]
torch_transforms = config["torch_transforms"]

model_weights = [model_weights_parser[list(model.keys())[0]] for model in torch_models]

if not models_from_path:
    torch_model_instances = []

    for model_instance, model_weight in zip(torch_models, model_weights):
        model_name = list(model_instance.keys())[0]
        model_code = model_instance[model_name]

        model_parse = f"models.{model_name}(weights=models.{model_weight}_Weights.IMAGENET1K_V1)"
        exec(f"{model_name} = {model_parse}")
        exec_code(model_code)

        torch_model_instances.append(eval(model_name))

torch_transforms = [eval(f"transforms.{transform}")(**torch_transforms[transform]) for transform in torch_transforms]

config["torch_transforms"] = torch_transforms
config["models"] = torch_model_instances

create_embeddings(**config)
# vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# del vgg16.classifier[-1]
# del vgg19.classifier[-1]

# del vgg16.classifier[-1]
# del vgg19.classifier[-1]

# del vgg16.classifier[-1]
# del vgg19.classifier[-1]


# models = [vgg16, vgg19]
# names = ["vgg16", "vgg19"]
# namespaces = ["embeddings/vgg16", "embeddings/vgg19"]
# batch_size = 16

# create_embeddings(models=models,
#                   run=image_gat_mp_run,
#                   namespaces=namespaces,
#                   data_path="../../benchmark/datasets/agadez/images",
#                   row_index_namespace="embeddings/row_index",
#                   torch_transforms=[transforms.Resize((512, 512)), 
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                                          std=[0.229, 0.224, 0.225])],
#                   batch_size=batch_size)
