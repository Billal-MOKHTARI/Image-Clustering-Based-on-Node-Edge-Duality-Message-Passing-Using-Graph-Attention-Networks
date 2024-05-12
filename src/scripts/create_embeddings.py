import os
import sys
from torchvision import transforms
import clip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tqdm import tqdm
from pipelines import create_embeddings
from torchvision import models
import json
from typing import Union, List
import argparse
from models.networks.constants import DEVICE

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

pretrained_model_weights_parser_path = "configs/pretrained_model_weights_parser.json"

with open(pretrained_model_weights_parser_path) as f:
    model_weights_parser = json.load(f)

torch_models = config["models"]
models_from_path = config["models_from_path"]
torch_transforms = config["torch_transforms"]

model_weights = [None for model in torch_models]
config["preprocess"] = [None for _ in torch_models]

if not models_from_path:
    torch_model_instances = []

    i = 0
    for model_instance, model_weight in zip(torch_models, model_weights):
        model_name = list(model_instance.keys())[0]       
        model_module = model_instance[model_name]["module"]
        
        if model_module == "torch":
            model_weights[i] = model_weights_parser[model_name]
            model_code = model_instance[model_name]["code"]
            model_parse = f"models.{model_name}(weights=models.{model_weight}_Weights.IMAGENET1K_V1)"
            exec(f"{model_name} = {model_parse}")
            exec_code(model_code)
            torch_model_instances.append(eval(model_name))
            config["preprocess"][i] = None
            
        elif model_module == "clip":
            model, preprocess = clip.load(model_name, device=DEVICE)
            torch_model_instances.append(model)
            config["preprocess"][i] = preprocess
        i += 1

if torch_transforms:
    config["torch_transforms"] = [eval(f"transforms.{transform}")(**torch_transforms[transform]) for transform in torch_transforms]

else:
    config["torch_transforms"] = None
    
config["models"] = torch_model_instances

create_embeddings(**config)