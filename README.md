'''
1. dataset/joint.py
Editing joint.py Adding masks to the target instances Modifying data augmentation

2. dataset/transforms.py
Modifying mask parts

3. main.py
Adding AppleMOTS data path
'''

'''
4. deformable_transformer_plus.py
Outputing the feature map from the transformer encoder (memoey) to use it as key for the MHAttention head
'''

'''
5. motr.py
5.1. Adding the segmentation head only in MOTR class
5.2. Adding segmentation postprocessing head in MORR class
5.3. Adding loss_masks to the ClipMatcher class and in get_loss function
5.4. Predicting masks in _forward_single_image function in MOTR class and adding it to frame_res
5.5. Adding mask to track instances in multiple functions including:
     5.5.1. _generate_empty_tracks function in MOTR class
     5.5.2. _post_process_single_image function in MOTR class
5.6. Adding pred_masks to outputs dictionary in forward function of the MOTR class
5.7. Evoking the pred_masks in match_for_single_frame function in ClipMatcher class
5.8. Calculating iou between masks:
     5.8.1. Initiate mask and box iou in _generate_empty_tracks function in MOTR class
     5.8.2. Calculating iou for box and mask in match_for_single_frame function in ClipMatcher class 
5.9. Adding pred_masks to TrackerPostProcess class 
5.10. Adding mask losses in weight_dict in build function
5.11. Including masks in losses list in build function
'''

'''
6. main.py
Adding args for masks in Loss coefficients
'''
