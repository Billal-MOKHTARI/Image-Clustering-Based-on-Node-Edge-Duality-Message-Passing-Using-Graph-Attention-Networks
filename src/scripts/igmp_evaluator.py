import argparse
import json
import os
import sys
from torch import nn
import torch
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.training.evaluator import igmp_evaluator
from models.networks import metrics

parser = argparse.ArgumentParser(prog='igmp_evaluator', description='Evaluate a Image GAT Message Passing model')

parser.add_argument('--config', type=str, help='Path to the configuration_file', required=True)

args = parser.parse_args()
config_path = args.config
 

with open(config_path, "r") as f:
    config = json.load(f)

use_custom_loss = config.get("use_custom_loss", False)

if "model_args" in config.keys():
    if not use_custom_loss:
        loss = eval(f"nn.{config['model_args']['loss']}")
        config["model_args"]["loss"] = loss
    else:
        loss = eval(f"metrics.{config['model_args']['loss']}")
        config["model_args"]["loss"] = loss


igmp_evaluator(**config)


