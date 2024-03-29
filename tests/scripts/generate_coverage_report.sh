#!/bin/bash

# Set the necessary environment variables
export REPORT_DIR="/workspaces/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/tests/reports"
export GENERATED_REPORT_DIR="/workspaces/Image-Clustering-Based-on-Node-Edge-Duality-Message-Passing-Using-Graph-Attention-Networks/htmlcov"

python -m pytest --cov=. --cov-report html
mv "${GENERATED_REPORT_DIR}" "${REPORT_DIR}" 

echo "Coverage report generated successfully!"