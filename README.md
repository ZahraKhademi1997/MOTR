# Project Updates Documentation

This document outlines the recent enhancements and modifications to the codebase, structured by files and specific functionalities. Each point details the nature of the changes and their impact on the corresponding modules.

## Dataset Modification
- `joint.py`: Adding masks to the target instances Modifying data augmentation
- `transforms.py`: Modifying mask parts

## Transformer Module Enhancements
- `deformable_transformer_plus.py` : Outputing the feature map from the transformer encoder (memoey) to use it as key for the MHAttention head

## MOTR Model Modifications
- `motr.py`:
1. Adding the segmentation head only in MOTR class
2. Adding segmentation postprocessing head in MOTR class
3. Adding loss_masks to the ClipMatcher class and in get_loss function
4. Predicting masks in _forward_single_image function in MOTR class and adding it to frame_res
5. Adding mask to track instances in multiple functions including:
     5.1. _generate_empty_tracks function in MOTR class
     5.2. _post_process_single_image function in MOTR class
6. Adding pred_masks to outputs dictionary in forward function of the MOTR class
7. Evoking the pred_masks in match_for_single_frame function in ClipMatcher class
8. Calculating iou between masks:
     8.1. Initiate mask and box iou in _generate_empty_tracks function in MOTR class
     8.2. Calculating iou for box and mask in match_for_single_frame function in ClipMatcher class 
9. Adding pred_masks to TrackerPostProcess class 
10. Adding mask losses in weight_dict in build function
11. Including masks in losses list in build function
12. Modifying MOTR class forward function to receive tensor data instead of dict
13. Creating dummy instance to contain the gt_instances inside the MOTR class forward function
14. Modifying ClipMatcher class forward function to receive only gt_data from the MOTR class forward function by creating dummy losses dict inside it
15. Modifying the MOTR class forward function to output dictionary of predictions instead of dictionary of two dictionaries of losses and predictions

## Postprocessing Modifications
- `qim.py`: Changing the iou in _select_active_tracks
- `segmentation.py`: Modifying PostProcessSegm function to incorporate in gradient

## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version

## Main Modifications
- `main.py`:
1. Adding AppleMOTS data path
2. Adding args for masks in Loss coefficients
3. Adding segmentation parameters in optimizer
4. Adding learning rates and losses to tensorboard
5. Adding AppleMOTS data path
6. Adding the add_graph to map model to tensorboard

## Execution Command
```bash 
configs/ sbatch --j model_graph_with_mask one_node_graph_segmentation.sh

```