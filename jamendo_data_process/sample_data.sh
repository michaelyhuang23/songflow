#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="../full_processed_jamendo_data"
DEST_DIR="../processed_jamendo_data"

# List of directories to copy
DIRS=("08" "58" "90" "50" "82")

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Loop through each directory and copy its contents
for i in "${!DIRS[@]}"; do
    dir=${DIRS[$i]}
    # Check if the directory exists
    if [ -d "$SOURCE_DIR/$dir" ]; then
        echo "Copying files from $SOURCE_DIR/$dir to $DEST_DIR"

        # Copy each file and rename it
        for file in "$SOURCE_DIR/$dir"/*; do
            filename=$(basename "$file")
            extension="${filename##*.}"
            new_filename="${i}.${extension}"
            cp "$file" "$DEST_DIR/$new_filename"
        done
    else
        echo "Directory $SOURCE_DIR/$dir does not exist"
    fi
done

echo "File copying and renaming complete."
