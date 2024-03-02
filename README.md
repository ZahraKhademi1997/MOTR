# Project Updates Documentation

This document outlines the recent enhancements and modifications to the codebase, structured by files and specific functionalities. Each point details the nature of the changes and their impact on the corresponding modules.

## Dataset MOdification
- `joint.py`: Adding masks to the target instances Modifying data augmentation
- `transforms.py`: Modifying mask parts

## Transformer Module Enhancements
- `deformable_transformer_plus.py` : Outputing the feature map from the transformer encoder (memoey) to use it as key for the MHAttention head

## MOTR Model Modifications
- `motr.py`:
1. Adding the segmentation head only in MOTR class
2. Adding segmentation postprocessing head in MORR class
3. Adding loss_masks to the ClipMatcher class and in get_loss function
4. Predicting masks in _forward_single_image function in MOTR class and adding it to frame_res
5. Adding mask to track instances in multiple functions including:
     5.1. _generate_empty_tracks function in MOTR class
     5.2. _post_process_single_image function in MOTR class
6. Adding pred_masks to outputs dictionary in forward function of the MOTR class
7. Evoking the pred_masks in match_for_single_frame function in ClipMatcher class
8. Calculating iou between masks:
     5.1. Initiate mask and box iou in _generate_empty_tracks function in MOTR class
     5.2. Calculating iou for box and mask in match_for_single_frame function in ClipMatcher class 
9. Adding pred_masks to TrackerPostProcess class 
10. Adding mask losses in weight_dict in build function
11. Including masks in losses list in build function

## Postprocessing Modifications
- `qim.py`: Changing the iou in _select_active_tracks
- `segmentation.py`: Modifying PostProcessSegm function to incorporate in gradient

## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version
- `engine.py`: Adding models gradients to tensorboard


## Main Modifications
- `main.py`:
1. Adding AppleMOTS data path
2. Adding args for masks in Loss coefficients
3. Adding segmentation parameters in optimizer
4. Adding learning rates and losses to tensorboard

## Execution Command
```bash 
configs/ sbatch --j applemots_train_mask one_node_mask_MOTR_train.sh

```