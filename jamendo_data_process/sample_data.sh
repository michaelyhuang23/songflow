#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="../full_processed_jamendo_data"
DEST_DIR="../processed_jamendo_data"

# List of directories to move
DIRS=("01" "53" "86" "38" "79")

# Loop through each directory and move its contents
for dir in "${DIRS[@]}"; do
    # Check if the directory exists
    if [ -d "$SOURCE_DIR/$dir" ]; then
        echo "Moving files from $SOURCE_DIR/$dir to $DEST_DIR"
        # Move the files
        mv "$SOURCE_DIR/$dir"/* "$DEST_DIR"/
    else
        echo "Directory $SOURCE_DIR/$dir does not exist"
    fi
done

echo "File move complete."

