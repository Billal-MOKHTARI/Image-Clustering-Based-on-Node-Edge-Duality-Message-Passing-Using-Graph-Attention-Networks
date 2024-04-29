import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from env import neptune_manager

parser = argparse.ArgumentParser(prog='delete_namespaces', description='Delete a namespace from a Neptune project\'s run')

parser.add_argument('--run', type=str, help='Name of the run', required=True)
parser.add_argument('--namespaces', nargs='+', type=str, help='Namespaces to delete', required=True)

args = parser.parse_args()

run = neptune_manager.Run(args.run)
run.delete_data(args.namespaces)