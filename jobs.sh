#!/bin/bash

python src/scripts/igmp_trainer.py --config /home/billalmokhtari/Documents/projects/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/configs/igmp_trainer.json --tloss
python src/scripts/delete_namespaces.py --run "Image GAT Message Passing" --namespaces training