#!/bin/bash

#SBATCH --cpus-per-task=20
#SBATCH -o sample_jamendo_data.txt
#SBATCH --job-name=sample_jamendo_data

SOURCE_DIR="../full_processed_jamendo_data"
DEST_DIR="../processed_jamendo_data"

DIRS=("08" "58" "90" "50" "82")

mkdir -p "$DEST_DIR"

for i in "${!DIRS[@]}"; do
    dir=${DIRS[$i]}
    if [ -d "$SOURCE_DIR/$dir" ]; then
        echo "Copying files from $SOURCE_DIR/$dir to $DEST_DIR"

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
