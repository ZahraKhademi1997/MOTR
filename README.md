# Project Updates Documentation

## TODO
- Using the whole queries in bbox_attention
- using conv instead of interpolation 
- using the masks before out layer in the mask head




This document outlines the recent enhancements and modifications to the codebase, structured by files and specific functionalities. Each point details the nature of the changes and their impact on the corresponding modules.

## Dataset Modification
- `gt_generation.py`: Generating gt.txt file from the mask images (Challenge: each object ids are extracted from the mask's pixel values, but there are some discrepancy in the pixel values within each frame which needs to be remapped to maintain continuous object ids)
- `prepare.py`: Adding function to create path files for Applemots
- `gen_labels_applemots.py`: To generate distinct label ext file for each frame and visualizing mask, bboxes and objesct ids on images
- `joint.py`: Adding masks to the target instances Modifying data augmentation
- `transforms.py`: Modifying mask parts
- `gen_mask_applemots.py`: Creating RLE mask for more efficient ID assigenment 

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
     5.1. Initiate mask and box iou in _generate_empty_tracks function in MOTR class
     5.2. Calculating iou for box and mask in match_for_single_frame function in ClipMatcher class 
9. Adding pred_masks to TrackerPostProcess class 
10. Adding mask losses in weight_dict in build function
11. Including masks in losses list in build function
12. Replacing pevious mask losses with dual focal loss and generalized dice loss.
13. Removing segmentation postprocessing frim the model and replace it with interpolation

## Association Modifications
- `matcher.py`: Adding the MIOU to the cost function

## Postprocessing Modifications
- `qim.py`: Changing the iou in _select_active_tracks
- `segmentation.py`: 
1. Modifying PostProcessSegm function to incorporate in gradient
2. Defining dual_focal_loss
3. Defining generlized_dice_loss
4. Replaing the out_lay in MaskHeadSmallConv with 1*1Conv and adding threshold --> back to 3*3 and adding relu after the last layer

## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version
- `engine.py`: Adding models gradients to tensorboard

## Utils Modifications
- `mask_ops.py`: Defining the mask iou 

## Main Modifications
- `main.py`:
1. Adding AppleMOTS data path
2. Adding args for masks in Loss coefficients
3. Adding args for masks in matcher coefficients
4. Adding segmentation parameters in optimizer
5. Adding learning rates to the optimizer

## Execution Command
```bash 
configs/ sbatch --j applemots_train_mask one_node_mask_MOTR_train.sh

```

### Data transfer instruction
```bash 
configs/ rsync -ah --progress -e 'ssh -v -p 22' home_directory username:hipergator directory
configs/ rsync -ah --progress -e 'ssh -p 22' username:/Hipergator directory /home directory
```
