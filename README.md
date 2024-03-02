# Project Updates Documentation

This document outlines the recent enhancements and modifications to the codebase, structured by files and specific functionalities. Each point details the nature of the changes and their impact on the corresponding modules.

## Dataset Modification
- `gt_generation.py`: Generating gt.txt file from the mask images (Challenge: each object ids are extracted from the mask's pixel values, but there are some discrepancy in the pixel values within each frame which needs to be remapped to maintain continuous object ids)
- `prepare.py`: Adding function to create path files for Applemots
- `gen_labels_applemots.py`: To generate distinct label ext file for each frame and visualizing mask, bboxes and objesct ids on images
- `joint.py`: Adding masks to the target instances Modifying data augmentation
- `transforms.py`: Modifying mask parts

## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version
- `engine.py`: Adding models gradients to tensorboard

## Main Modifications
- `main.py`:
1. Adding AppleMOTS data path
2. Adding learning rates and losses to tensorboard

## Evaluation script
- `apple_eval.py`: Writing apple_eval.py for applemots evaluation

## Execution Command
```bash 
configs/ sbatch --j applemots_motr_bbox applemots_one_node_train.sh

```