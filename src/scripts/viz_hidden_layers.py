import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pipelines import viz_hidden_layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
import json
from tqdm import tqdm
from torchvision import models, transforms

parser = argparse.ArgumentParser(prog='viz_hidden_layers', description='Visualize hidden layers\' outputs')
parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)

args = parser.parse_args()
path = args.config

# read configuration file
with open(path) as f:
    config = json.load(f)

pretrained_model_weights_parser_path = "../../configs/pretrained_model_weights_parser.json"

with open(pretrained_model_weights_parser_path) as f:
    model_weights_parser = json.load(f)


torch_models = config["models"]
namespaces = config["namespaces"]
models_from_path = config["models_from_path"]
torch_transforms = config["torch_transforms"]
image_path = config["image_path"]
run = config["run"]
model_weights = [model_weights_parser[model] for model in torch_models]
method = config["method"]


# parse the parameters
if not models_from_path:
    torch_models = [eval(f"models.{model}(weights=models.{model_weight}_Weights.IMAGENET1K_V1)") for model, model_weight in tqdm(zip(torch_models, model_weights), total=len(torch_models), desc="Loading models", colour="green")]

torch_transforms = [eval(f"transforms.{transform}")(**torch_transforms[transform]) for transform in torch_transforms]


viz_hidden_layers(models = torch_models, 
                  image_path=image_path, 
                  run=run, 
                  namespaces=namespaces, 
                  torch_transforms=torch_transforms,
                  method=method,
                  models_from_path=models_from_path)

