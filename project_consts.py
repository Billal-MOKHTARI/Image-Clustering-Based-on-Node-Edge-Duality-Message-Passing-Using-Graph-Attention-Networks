from torch_model_manager import NeptuneManager
import json
import os

# Create a ConfigParser object
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path) as f:
    config = json.load(f)

neptune_manager_configs = config["neptune_configs"]
project = neptune_manager_configs["project"]
api_token = neptune_manager_configs["api_token"]
run_ids_path = neptune_manager_configs["run_ids_path"] 
run_ids_path = os.path.join(os.path.dirname(__file__), run_ids_path)

NEPTUNE_MANAGER = NeptuneManager(project=project,
                    api_token=api_token,
                    run_ids_path=run_ids_path)


# List of runs
DATA_VISUALIZATION_RUN = NEPTUNE_MANAGER.create_run("data_visualization")