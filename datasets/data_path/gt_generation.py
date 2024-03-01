import os 
import numpy as np 
from PIL import Image
import pandas as pd 

############################################################## Before Remapping IDs ###############################################################
def overview(train_dir, bbox_save_dir):
    dirs = [train_dir]
    dicts = {}
    
    for folder in dirs:
        path = os.path.join(folder, 'instances')
        for dataset in os.listdir(path):
            print(dataset)
            mask, track, frames = 0, 0, 0

            for filename in os.listdir(os.path.join(path, dataset)):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    frames += 1
                    img = np.array(Image.open(os.path.join(path, dataset, filename)))
                    obj_ids = np.unique(img)[1:]  # Exclude background
                    
                    mask += len(obj_ids)
                    
                    # Extract and log bounding boxes with track IDs
                    with open(os.path.join(bbox_save_dir, f'{dataset}_gt.txt'), 'a') as f:
                        for obj_id in obj_ids:
                            obj_mask = (img == obj_id).astype(np.uint8)
                            pos = np.where(obj_mask)
                            xmin = np.min(pos[1]) 
                            xmax = np.max(pos[1])
                            ymin = np.min(pos[0]) 
                            ymax = np.max(pos[0]) 
                            w = xmax - xmin
                            h = ymax - ymin
                            object_id = (obj_id%1000)+1
                            frame_id = int(filename.split(".")[0]) + 1
                            # print(frame_id)
                            # f.write(f'{filename},{obj_id},{xmin},{ymin},{xmax},{ymax}\n')
                            f.write(f'{frame_id},{object_id},{xmin},{ymin},{w},{h}, -1, -1, -1, -1\n')


                    if np.max(obj_ids) > track:
                        track = np.max(obj_ids)
                else:
                    continue
            track_id = track % 1000
            ls = [frames, track_id, mask]
            dicts[dataset] = ls

    df = pd.DataFrame.from_dict(dicts, orient='index', columns=['#frames', '#tracks', '#masks'])
    return df



# train_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/"
# test_dir="/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/test/"
# bbox_save_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt"

# overview(train_dir,bbox_save_dir)


############################################################## After Remapping IDs ###############################################################
import os
import pandas as pd   
from collections import defaultdict

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Parse the lines into a list of lists
    data = [line.strip().split(',') for line in lines]
    return data



# Read the file and take a look at the first few entries to determine the structure
file_path = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt/gt_before_remapping/0005_gt.txt"

def adjust_ids_continuously(data_by_frame):
    global_id_counter = 0
    id_mapping = {}
    adjusted_data_by_frame = defaultdict(list)

    for frame_id, frame_data in sorted(data_by_frame.items()):
        local_ids = set()
        for entry in frame_data:
            obj_id = int(entry[1])
            if obj_id not in id_mapping:
                id_mapping[obj_id] = global_id_counter = global_id_counter + 1
            local_ids.add(id_mapping[obj_id])
            entry[1] = str(id_mapping[obj_id])
            adjusted_data_by_frame[frame_id].append(entry)
        # Ensure continuity
        max_local_id = max(local_ids) if local_ids else 0
        if global_id_counter < max_local_id:
            global_id_counter = max_local_id

    return [item for sublist in adjusted_data_by_frame.values() for item in sublist]

# # Read the data from the file
# tracking_data = read_data(file_path)

# # Organize data by frame ID
# data_by_frame = defaultdict(list)
# for entry in tracking_data:
#     frame_id = int(entry[0])
#     data_by_frame[frame_id].append(entry)

# # Adjust object IDs to be continuous within each frame and consistent across all frames
# adjusted_data = adjust_ids_continuously(data_by_frame)

# # Write the adjusted data to a new text file
# output_file_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt/gt_after_remapping/0005_gt.txt'
# with open(output_file_path, 'w') as file:
#     for entry in adjusted_data:
#         file.write(','.join(entry) + '\n')



############################################################## TXT FILE VISUALIZATION ###############################################################
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to the directory containing images
images_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/images/0005"
gt_file = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt/gt_after_remapping/0005_gt.txt"
output_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt/vis/image5"

def plot_bboxes_and_ids(gt_file_path, images_dir, output_dir):
    # Use a dictionary to accumulate bbox data for each frame
    frame_data = defaultdict(list)

    with open(gt_file_path, 'r') as file:
        lines = file.readlines()

    # Group bbox data by frame_id
    for line in lines:
        frame_id, obj_id, xmin, ymin, width, height = map(int, line.split(',')[:6])
        frame_data[frame_id].append((obj_id, xmin, ymin, width, height))

    # Plot all bboxes for each frame
    for frame_id, bboxes in frame_data.items():
        image_name = f"{frame_id-1:06d}.png"
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)

        for obj_id, xmin, ymin, width, height in bboxes:
            cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
            cv2.putText(image, str(obj_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(os.path.join(output_dir, image_name), image)
        
        
plot_bboxes_and_ids(gt_file, images_dir, output_dir)