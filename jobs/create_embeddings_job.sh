#!/bin/bash
# We suppose that we are in jobs folder


cd ../src/scripts

# Find all JSON files inside the igmp_trainer directory
json_files=$(find ../../configs/embeddings -type f -name "*.json")
# Iterate over each JSON file
for file in $json_files; do
    # Execute the command for each file
    python create_embeddings.py --config "$file"
done