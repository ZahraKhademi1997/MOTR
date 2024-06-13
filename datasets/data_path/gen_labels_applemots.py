###################################################################################################### STEP1: CREATING TXT FILES WITH CONTINUOUS IDS FROM THE GT.TXT (SUCCESSFUL) ######################################################################################################
import os
import os.path as osp
import numpy as np 
from PIL import Image
import pandas as pd 



############################################################## Creating gt.txt Before Remapping IDs ###############################################################
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



train_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/"
bbox_save_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/gt_files/"

overview(train_dir,bbox_save_dir)




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

###################################################################################################### STEP1: CREATING TXT FILES WITH CONTINUOUS IDS FROM THE GT.TXT (SUCCESSFUL) ######################################################################################################

seqs = ['0006', '0007', '0008', '0010', '0011', '0012']

seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/testing/images/'
label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/testing/labels_with_ids/'


# Function to create directories if they don't exist
def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)

tid_curr = 0
tid_last = -1

for seq in seqs:
    # Adapt these lines to match how you retrieve sequence information, such as image width and height
    # seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(1296)
    seq_height = int(972)

    # Load the ground truth data; adjust the path and format as needed
    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    idx = np.lexsort(gt.T[:2, :])
    gt = gt[idx, :]

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    # Initialize a dictionary to keep track of written object IDs for each frame
    # written_obj_ids_per_frame = {}
    for fid, tid, x, y, w, h, mark, _, _, _ in gt:
        if mark == 0:
            continue
        filename_fid = int(fid) - 1
        fid = int(fid) 
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        # Calculate center x, y
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(filename_fid))
        # print('label_fpath is:', label_fpath)
        # Update the format of the label string if necessary
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:d}\n'.format(
            fid, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height, tid)
        with open(label_fpath, 'a') as f:
            # print('label_fpath is:', label_fpath)
            f.write(label_str)
            

