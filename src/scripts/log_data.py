import pandas as pd
import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from env import neptune_manager

argparser = argparse.ArgumentParser(prog='log_data', description='Log the data to Neptune')

argparser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
args = argparser.parse_args()
config_path = args.config

with open(config_path, 'r') as f:
    config = json.load(f)
    
allowed_image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'svg']

for key, value in config.items():
    if os.path.isfile(key):
        if key.endswith('.csv'):
            df = pd.read_csv(key, **value["read_csv_args"])
            neptune_manager.log_dataframe(dataframe=df, **value["log_dataframe_args"])
        else:
            neptune_manager.log_files(data=None, 
                                      namespace=value["namespace"],
                                      from_path=value["from_path"],
                                      extension=key.split(".")[-1],
                                      wait=value["wait"])
    elif os.path.isdir(key):
        files = [os.path.join(key, f) for f in os.listdir(key) if f.lower().endswith(tuple(allowed_image_extensions))]
        for file in files:
            neptune_manager.log_files(data=None, 
                                      namespace=f'{os.path.join(value["namespace"], file.split("/")[-1].split(".")[0])}',
                                      from_path=file,
                                      extension=file.split(".")[-1],
                                      wait=value["wait"])
    