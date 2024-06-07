import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env import neptune_manager
import argparse

argparser = argparse.ArgumentParser(prog='download_folder_from_neptune', description='Downlaod folder from Neptune')
argparser.add_argument('--run', 
                       type=str, 
                       help='Neptune run name. If it is not specified, it will take the neptune project namespace by default.', required=False)
argparser.add_argument('--namespace', type=str, help='Neptune folder namespace', required=True)
argparser.add_argument('--path', type=str, help='Path to save the folder', required=True)

args = argparser.parse_args()

if args.run is not None:
    run = neptune_manager.Run(args.run)
else:
    run = neptune_manager
    
run.download_folder_from_neptune(args.namespace, args.path)