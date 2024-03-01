'''

1. datasets.datapath
1.1. gt_generation.py
Generating gt.txt file from the mask images 
--> content: frame_id, objcet_id, xmin, ymin, w, h, -1 , -1, -1, -1
--> Challenge: each object ids are extracted from the mask's pixel values, but there are some discrepancy in the pixel values within each frame, for instance in frame 000000.png in seq "0", I have obj_ids from 1 to 104 and then jumps to 228 and so on. To solve this challenge, I remapped the extracted ids to stay continuous in each frame.

1.2. prepare.py
Adding function to create path files for Applemots

'''