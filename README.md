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
- `deformable_transformer_plus.py` : 
1. Adding seg_encoder to encode the feature map from the encoder along with the position and spatial shape and reference points at the first level
2. Returning the (seg_memory, seg_pos_embed, seg_mask, spatial_shapes[[0]], seg_reference_points, level_start_index[0], valid_ratios[:, [0], :]) for further processing of the mask in MOTR.

## MOTR Model Modifications
- `motr.py`:
1. Adding seg_branch in MOTR to embed the queries from the transformer.decoder
2. Adding mask_positional_encoding to integrate the location information of each object queries
3. Adding dynamic_encoder to only attand on a specific instance using the attention mechanism
4. Adding loss_masks to the ClipMatcher class and in get_loss function
5. Predicting masks in _forward_single_image function in MOTR class and adding it to frame_res
6. Adding mask to track instances in multiple functions including:
     6.1. _generate_empty_tracks function in MOTR class
     6.2. _post_process_single_image function in MOTR class
7. Adding pred_masks to outputs dictionary in forward function of the MOTR class
8. Evoking the pred_masks in match_for_single_frame function in ClipMatcher class
9. Adding pred_masks to TrackerPostProcess class 
10. Adding mask losses in weight_dict in build function
11. Including masks in losses list in build function
12. Setting the num_class from 1 to 2 to include the background


## Association Modifications
- `matcher.py`: 
1. Adding focal loss and dice loss to the cost function

## Postprocessing Modifications
- `qim.py`: Changing the iou in _select_active_tracks
- `segmentation.py`: 
1. Add aligned_bilinear function for mask postprocessing

## Tools Modifications
- `misc.py`: Modifying import part to adjust for the current torchvision version
- `engine.py`: Adding models gradients to tensorboard

## Utils Modifications
- `mask_ops.py`: 
1. Adding dice loss for matcher
2. Adding focal loss for matcher

## MMCV_utils
- `dynamicdeformableattention.py`: To define the DynamicDeformableAttention as dynamic_encoder
- `mask_position_encoding.py`: To define the the RelSinePositionalEncoding
- `transformerLayerSequence.py`: To define TransformerLayerSequence as seg_encoder

## Main Modifications
- `main.py`:
1. Adding AppleMOTS data path
2. Adding args for masks in Loss coefficients
3. Adding args for masks in matcher coefficients
4. Adding segmentation parameters in optimizer
5. Adding learning rates to the optimizer

## Environment Configs
- `conda config`:
```
conda create -n MOTR-instance
```

```
conda activate MOTR-instance
```

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

```
pip install -r requirements.txt
```

```bash 
configs/ sbatch --j configuration slurm_conda_env_config.sh
```

- `Internimage config`:
```
pip install -U openmim
```

```
mim install mmcv (2.2.0)
```

```
mim install mmcv-full (1.7.0)
```

```
pip install timm==0.6.11 mmdet==2.28.1
```




## Execution Command
```bash 
configs/ sbatch --j applemots_train_mask one_node_slurm.sh

```

### Data transfer instruction
```bash 
configs/ rsync -ah --progress -e 'ssh -v -p 22' home_directory username:hipergator directory
configs/ rsync -ah --progress -e 'ssh -p 22' username:/Hipergator directory /home directory
```
