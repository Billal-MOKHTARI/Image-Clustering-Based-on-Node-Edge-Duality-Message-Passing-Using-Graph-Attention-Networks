import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# git config --global core.excludesfile ~/.gitignore
# python src/scripts/delete_namespaces.py --namespaces data/images/annotated_images_geometric data/csv_files/annotations_geometric data/detections data/tensors
# python src/scripts/create_adjacency_tensor.py --config configs/create_adjacency_tensor_test.json
# bash src/scripts/remove_duplicates.sh benchmark/datasets/agadez/classes_1.txt benchmark/datasets/agadez/classes_2.txt


# from env import neptune_manager
# import subprocess
# import re

# files = neptune_manager.fetch_files("data/detections/detections_agadez")


# def extract_files_with_extensions(strings):
#     # Regular expression to match strings with file extensions
#     pattern = re.compile(r'\.\w+$')
#     return [s for s in strings if pattern.search(s)]


# files_with_extensions = extract_files_with_extensions(files)

# paths = [f"data/detections/detections_agadez/{f}" for f in files_with_extensions]

# for path in paths:
#     subprocess.run(["python", "src/scripts/delete_namespaces.py", "--namespaces", path])
# print(len(files))