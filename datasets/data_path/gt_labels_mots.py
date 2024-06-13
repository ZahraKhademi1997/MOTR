import os 
import numpy as np 
from PIL import Image
import pandas as pd 
import os 
import os.path as osp
import numpy as np 
from PIL import Image
import pandas as pd 
import pycocotools.mask as mask_utils
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pycocotools import mask as mask_utils
from PIL import Image

############################################################## Creating gt.txt Before Remapping IDs ###############################################################
def calculate_bbox_from_mask(mask, img_height, img_width):
    if mask.max() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax - xmin, ymax - ymin

# For masks
def decode_RLE_to_mask(rle_str, h, w):
    rle = {
        'counts': rle_str,
        'size': [h, w]
    }
    mask = mask_utils.decode(rle)
    return mask

def overview(train_dir, bbox_save_dir):
    track_id_dict = {}  # Dictionary to maintain unique object tracking across frames
    dirs = [train_dir]
    dicts = {}
    
    for folder in dirs:
        path = os.path.join(folder, 'gt')
        for dataset in os.listdir(path):
            mask, track, frames = 0, 0, 0
            if not os.path.exists(bbox_save_dir):
                os.makedirs(bbox_save_dir)
            
            output_file_path = os.path.join(bbox_save_dir, f'{dataset}.txt')
            
            with open(os.path.join(path, dataset), 'r') as file, open(output_file_path, 'w') as out_file:
                for line in file:
                    data = line.strip().split()
                    frame_id, object_id, img_height, img_width, rle = int(data[0]), int(data[1]), int(data[3]), int(data[4]), data[5]
                    object_key = (dataset, object_id)
                    
                    if object_key not in track_id_dict:
                        track_id_dict[object_key] = len(track_id_dict) + 1  # Assign new track id
                    
                    binary_mask = decode_RLE_to_mask(rle, img_height, img_width)
                    bbox = calculate_bbox_from_mask(binary_mask, img_height, img_width)
                    if bbox:
                        xmin, ymin, width, height = bbox
                        # Write the reformatted data to the output file
                        out_file.write(f"{frame_id},{track_id_dict[object_key]},{img_width},{img_height},{rle}\n")
                        mask += 1
                        if frame_id > frames:
                            frames = frame_id
                        if track_id_dict[object_key] > track:
                            track = track_id_dict[object_key]
            
            dicts[dataset] = {'#frames': frames, '#tracks': track, '#masks': mask}
    
    df = pd.DataFrame.from_dict(dicts, orient='index', columns=['#frames', '#tracks', '#masks'])
    return df

train_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/images/MOTS20-02/"
bbox_save_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/gt_files/"

# overview(train_dir,box_save_dir)



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
# ################################################## For bboxes ###############################################
seqs = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']

seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/images/'
label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/labels_with_ids/'


# Function to create directories if they don't exist
def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)

tid_curr = 0
tid_last = -1

for seq in seqs:
    # Adapt these lines to match how you retrieve sequence information, such as image width and height
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

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
            
# ################################################## For masks ###############################################

seqs = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
# seqs = ['MOTS20-02']
seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/images/'
label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/images/masks_with_ids/'
info_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/MOTS/train/images/'

# Function to create directories if they don't exist
def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)

tid_offset = 0
for seq in seqs:
    # Adapt these lines to match how you retrieve sequence information, such as image width and height
    seq_info = open(osp.join(info_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    #     # Load the ground truth data; adjust the path and format as needed
    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    
    col_names = ['frame_id', 'track_id', 'w', 'h', 'rle']
    # col_types = {'frame_id': int, 'track_id': int, 'x': float, 'y': float, 'w': float, 'h': float, 'mark': int, 'rle': str}

    # gt = pd.read_csv(gt_txt, header=None, names=col_names, dtype=col_types)
    gt = pd.read_csv(gt_txt, header=None, names=col_names)
    gt.sort_values(by=['frame_id', 'track_id'], inplace=True)
    # if gt['track_id'].all() == 79:
    #     print(gt['track_id'])
    # output = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/output/id.txt'
    # with open(output, 'w') as f:
    #     f.write(str(gt['track_id']))
    max_tid_in_seq = gt['track_id'].max() if not gt.empty else 0
    print('max_tid_in_seq:', max_tid_in_seq)

    for index, row in gt.iterrows():
        
        fid, tid, img_width , img_height, rle = row
        class_id = 1
        class_id = int(class_id)
        img_width = int(img_width) 
        img_height = int(img_height)
        
        # if tid == 79:
        #     print(f"Track ID 79 found in {seq} at frame {fid}")

        filename_fid = int(fid)
        fid = int(fid) 
        tid = int(tid)
        tid += tid_offset
        # if not tid == tid_last:
        #     tid_curr += 1
        #     tid_last = tid
        # Calculate center x, y
        
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(filename_fid))
        # print('label_fpath is:', label_fpath)
        # Update the format of the label string if necessary
        label_str = '{:d} {:d} {:d} {:d} {:d} {}\n'.format(
            fid, tid, int(class_id), img_width , img_height , rle)
        with open(label_fpath, 'a') as f:
            # print('label_fpath is:', label_fpath)
            f.write(label_str)
    tid_offset += max_tid_in_seq 




