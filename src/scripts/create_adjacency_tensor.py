import argparse
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pipelines import create_adjacency_tensor

argparser = argparse.ArgumentParser(prog='create_adjacency_tensor', description='Create an adjacency tensor from an annotation matrix')
argparser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
args = argparser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

adjacency_tensor = create_adjacency_tensor(**config)
print(adjacency_tensor)