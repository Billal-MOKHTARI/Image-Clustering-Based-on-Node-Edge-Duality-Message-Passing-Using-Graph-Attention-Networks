#!/bin/bash

# Function to compare files in two folders
compare_folders() {
    folder1="$1"
    folder2="$2"
    
    identical_files=0
    total_files=0

    # Iterate over files in folder1
    for file1 in "$folder1"/*; do
        # Extract filename
        filename=$(basename "$file1")

        # Check if corresponding file exists in folder2
        if [ -f "$folder2/$filename" ]; then
            # Increment total files counter
            ((total_files++))

            # Compare files
            if cmp -s "$file1" "$folder2/$filename"; then
                ((identical_files++))
                echo "$filename: the files in $folder1 and $folder2 are identical."
            else
                echo "$filename: the files in $folder1 and $folder2 are different."
            fi
        else
            echo "$filename: $folder2/$filename does not exist."
        fi
    done

    # Calculate percentage of identical files
    if [ "$total_files" -ne 0 ]; then
        percentage=$((100 * identical_files / total_files))
        echo "Percentage of identical files: $percentage%"
    else
        echo "No files to compare."
    fi
}

# Main script starts here

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder1> <folder2>"
    exit 1
fi

folder1="$1"
folder2="$2"

# Check if both folders exist
if [ ! -d "$folder1" ] || [ ! -d "$folder2" ]; then
    echo "Error: One or both folders do not exist."
    exit 1
fi

# Compare files in the folders
compare_folders "$folder1" "$folder2"
