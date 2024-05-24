import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# git config --global core.excludesfile ~/.gitignore
# python src/scripts/delete_namespaces.py --namespaces data/images/annotated_images_geometric data/csv_files/annotations_geometric data/detections data/tensors
# python src/scripts/create_adjacency_tensor.py --config configs/create_adjacency_tensor_test.json
# bash src/scripts/remove_duplicates.sh benchmark/datasets/agadez/classes_1.txt benchmark/datasets/agadez/classes_2.txt