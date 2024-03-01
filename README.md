'''
1. datasets.datapath
1.1. Writing gt_generation.py
Generating gt.txt file from the mask images 
--> content: frame_id, objcet_id, xmin, ymin, w, h, -1 , -1, -1, -1
--> Challenge: each object ids are extracted from the mask's pixel values, but there are some discrepancy in the pixel values within each frame, for instance in frame 000000.png in seq "0", I have obj_ids from 1 to 104 and then jumps to 228 and so on. To solve this challenge, I remapped the extracted ids to stay continuous in each frame.

1.2.  prepare.py
Adding function to create path files for Applemots

1.3. Writing gen_labes_applemots.py
To generate distinct label ext file for each frame and visualizing mask, bboxes and objesct ids on images

2. dataset
2.1. Editing joint.py
Adding masks to the target instances
Modifying data augmentation


3. main.py
Adding AppleMOTS data path

4. transforms.py
Modifying mask parts

'''


'''
5. engine.py
Logging gradients to tensorboard

6. main.py
Logging learnign rates to tensorboard
'''

'''
7. configs
Adding applemots_one_node_train.sh for training on Hipergator
'''

'''
8. Writing apple_eval.py
Writing apple_eval.py for applemots evaluation

'''