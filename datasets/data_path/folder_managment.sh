#!/bin/bash


DIR_IN=mask_segmentation_py
DIR_OUT=visual_folders
MAX_FILES=300
SUBDIR_PREFIX="subdir"
COUNTER=1

# Create a directory to hold subdirectories if it doesn't exist
mkdir -p "${DIR_OUT}/subdirectories"

# Move files into subdirectories
find "$DIR_IN" -maxdepth 1 -type f | while read FILE; do
    if [ -f "${FILE}" ]; then # Check if it's a file
        # Create new subdirectory if needed
        if [ $((COUNTER % MAX_FILES)) -eq 1 ]; then
            SUBDIR="${DIR_OUT}/subdirectories/${SUBDIR_PREFIX}_$((COUNTER / MAX_FILES + 1))"
            mkdir -p "${SUBDIR}"
            echo "One folder done"
        fi
        
        # Move the file
        cp "${FILE}" "${SUBDIR}/"
        let COUNTER=COUNTER+1
    fi
done

echo "Files are distributed into subdirectories."