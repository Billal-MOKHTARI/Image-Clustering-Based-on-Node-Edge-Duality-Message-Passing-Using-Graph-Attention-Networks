#!/bin/bash

cd ../src/scripts

python delete_namespaces.py --namespaces data figures
python create_embeddings.py --config ../../configs/create_embeddings.json
python viz_hidden_layers.py --config ../../configs/viz_hidden_layers.json