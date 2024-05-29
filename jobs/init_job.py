import argparse
import json
import subprocess

argparser = argparse.ArgumentParser(prog='init_job', description='Initialize the project')
argparser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
args = argparser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

import subprocess

def execute_commands(config):
    # Sort the dictionary based on the exec_order
    sorted_data = dict(sorted(config.items(), key=lambda item: item[1]['exec_order']))
    
    # Loop through the sorted dictionary and execute the commands
    for key, value in sorted_data.items():
        if value.get('exec', False):
            # Construct the base command
            script = key
            command = f"python src/scripts/{script}.py"
            
            # Add parameters from the sub-dictionary to the command
            for param, param_value in value.items():
                if param not in ['exec', 'exec_order']:  # Skip exec and exec_order keys
                    command += f" --{param} {param_value}"
            
            print(f"Executing: {command}")
            try:
                result = subprocess.run(command, shell=True, check=True)
                if result.returncode != 0:
                    print(f"Command failed: {command}")
                    break
            except subprocess.CalledProcessError as e:
                print(f"Command '{command}' failed with error: {e}")
                break

execute_commands(config)