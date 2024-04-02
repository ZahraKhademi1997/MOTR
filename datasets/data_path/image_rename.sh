#!/bin/bash

# Directory containing the original images
SOURCE_DIR=/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/dice_loss_py

# Directory to store the renamed images
DEST_DIR=/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/dice_loss_py_renamed_6

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Initialize the counter
counter=95280

# Change to the source directory
cd "$SOURCE_DIR"

# Loop through all png files that match the pattern
for file in mask_comparison_ep_idx1_*.png; do
    # Format the new filename with the counter as the number
    newname="mask_comparison_ep_idx1_${counter}.png"
    
    # Move and rename the file to the new directory
    mv "$file" "${DEST_DIR}/${newname}"
    
    # Increment the counter
    ((counter++))
done


# chmod +x image_rename.sh
# ./image_rename.sh 