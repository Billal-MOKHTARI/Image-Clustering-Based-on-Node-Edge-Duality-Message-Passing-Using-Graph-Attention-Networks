#!/bin/bash

# Author: [Billal MOKHTARI]
# Date: [07/06/2024]

python src/scripts/download_folder_from_neptune.py --namespace data/detections/detections_agadez --path benchmark/datasets/agadez/detections
python src/scripts/download_folder_from_neptune.py --namespace data/images/annotated_images_agadez --path benchmark/datasets/bamako/detections