# Project Updates Documentation
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

## Backbone Modification
- `joiner`: Adding output_shape function to create a dictionary of resnet different outputs (layer 2, layer 3, layer 4).

## MOTR Model Modifications
- `motr.py`:
1. Adding AxialAttention, CrossAttention, segmentation brach and PerPixelEmbedding in MOTR class
2. Adding loss_masks to the ClipMatcher class and in get_loss function and AUX-losses for the multi-scale embedded queries:
     2.1. Global Loss: The DiceLoss and CrossEntropyLoss between the values at sampled points and groundtruth 
3. Predicting masks in _forward_single_image function in MOTR class and adding it to frame_res
     3.1. Embedding feature maps from backbone using PerPixelEmbedding function
     3.2. Embedding multi-scale queries using the segmentation branch 
     3.3. Performing cross attention between embedded queries and perpixelembedding
     3.4. Applying axial attention to the perpixel embedding
     3.5. Performing matrix multipliction between output of the cross attention and the output of axial attention
4. Adding mask to track instances in multiple functions including:
     5.1. _generate_empty_tracks function in MOTR class
     5.2. _post_process_single_image function in MOTR class
5. Adding pred_masks to outputs dictionary in forward function of the MOTR class
6. Evoking the pred_masks in match_for_single_frame function in ClipMatcher class
7. Adding pred_masks to TrackerPostProcess class 
8. Adding mask losses and aux_mask_losses in weight_dict in build function
9. Including masks in losses list in build function


## Association Modifications
- `matcher.py`: Adding the Focal Loss and Dice Loss to the totall cost function

## Postprocessing Modifications
- `qim.py`: Changing the iou in _select_active_tracks
- `segmentation.py`: 
1. Defining cross entropy loss for masks


## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version
- `engine.py`: Adding models gradients to tensorboard

## Utils Modifications
- `mask_ops.py`: Defining the mask iou 
- `shapespace.py`, Defing outputshape for backbone
- `PerPixelEmbedding.py` , Defining FPN structure for upsampling the feature map from the backbone using laterla convolution
- `axial_attention.py` , Defining the axial attention function to perform attention in both axis independently
- `cross_attention.py` , Defining the cross attention function to perform the cross attention between the output of the PerPixelEmbedding and learned queries
- `points.py` , This script containes the following functions:
               1.1. `get_uncertainty` : To calculate the high uncertain area in which exist in predicted mask but not in groundtruth mask
               1.2. `get_uncertain_point_coords_with_randomness` : Samples points across the predicted mask. Majority of the sampled points are from the highly uncertain areas.
               1.3. `point_sample` : Extracts values in the sampled-point areas.

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
