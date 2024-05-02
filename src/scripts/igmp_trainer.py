import argparse
import json
import os
import sys
from torch import nn
import torch
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.training.trainer import image_gat_mp_trainer as igmp_trainer

parser = argparse.ArgumentParser(prog='igmp_trainer', description='Train a Image GAT Message Passing model')

parser.add_argument('--config', type=str, help='Path to the configuration_file', required=True)
parser.add_argument('--tloss', action='store_true', help='If this argument is given, that means that the loss used is a pytorch built-in module.', required=False)

args = parser.parse_args()
config_path = args.config
tloss = args.tloss

with open(config_path, "r") as f:
    config = json.load(f)

if "model_args" in config.keys():
    if tloss:
        loss = eval(f"nn.{config['model_args']['loss']}")
        config["model_args"]["loss"] = loss

optim = eval(f"torch.optim.{config['optimizer']}")
config["optimizer"] = optim

igmp_trainer(**config)


