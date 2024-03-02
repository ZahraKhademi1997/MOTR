'''
1. dataset/joint.py
Editing joint.py Adding masks to the target instances Modifying data augmentation

2. dataset/transforms.py
Modifying mask parts

3. main.py
Adding AppleMOTS data path

4. motr.py
4.1. Modifying MOTR class forward function to receive tensor data instead of dict
4.2. Creating dummy instance to contain the gt_instances inside the MOTR class forward function
4.3. Modifying ClipMatcher class forward function to receive only gt_data from the MOTR class forward function by creating dummy losses dict inside it
4.4. Modifying the MOTR class forward function to output dictionary of predictions instead of dictionary of two dictionaries of losses and predictions


5. main.py
Adding the add_graph to map model to tensorboard

'''
