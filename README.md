# Project Updates Documentation

This document outlines the recent enhancements and modifications to the codebase, structured by files and specific functionalities. Each point details the nature of the changes and their impact on the corresponding modules.

## Dataset Modification
- `joint.py`: Adding masks to the target instances Modifying data augmentation
- `transforms.py`: Modifying mask parts

## MOTR Model Modifications
- `motr.py`:
1. Modifying MOTR class forward function to receive tensor data instead of dict
2. Creating dummy instance to contain the gt_instances inside the MOTR class forward function
3. Modifying ClipMatcher class forward function to receive only gt_data from the MOTR class forward function by creating dummy losses dict inside it
4. Modifying the MOTR class forward function to output dictionary of predictions instead of dictionary of two dictionaries of losses and predictions

## Main Modifications
- `main.py`:
1. Adding AppleMOTS data path
2. Adding the add_graph to map model to tensorboard

## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version

## Execution Command
```bash 
configs/ sbatch --j model_graph_without_mask one_node_graph_without_mask.sh

```