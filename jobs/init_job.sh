#!/bin/bash

python src/scripts/delete_namespaces.py --namespaces data figures
python src/scripts/create_embeddings.py --config configs/create_embeddings.json
python src/scripts/viz_hidden_layers.py --config configs/viz_hidden_layers.json
python src/scripts/log_data.py --config configs/log_data.json