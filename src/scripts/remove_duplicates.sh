#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 input_file [output_file]"
    exit 1
fi

# Set input and output file names
input_file="$1"
output_file="${2:-output.txt}"

# Remove duplicate lines and preserve the order
awk '!seen[$0]++' "$input_file" > "$output_file"

echo "Duplicates removed and saved to $output_file"
