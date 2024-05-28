import argparse
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env import neptune_manager
from models.networks.constants import DEVICE 
from torch_model_manager import SegmentationManager


argparser = argparse.ArgumentParser(prog='generate_masks', description='generate masks from SAM')
argparser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
args = argparser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)
run = config.get("run", None)

if run is None:
    run = neptune_manager
else:
    run = neptune_manager.Run(run)
config["run"] = run

seg_manager = SegmentationManager(device=DEVICE)
seg_manager.auto_seg_images(**config)