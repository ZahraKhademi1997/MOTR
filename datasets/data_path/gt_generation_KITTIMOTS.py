import os 
import os.path as osp
import numpy as np 
from PIL import Image
import pandas as pd 
import pycocotools.mask as mask_utils
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

############################################################## STEP1: Creating gt.txt  ###############################################################
# MOTS
import os
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image

# def decode_RLE_to_mask(rle, height, width):
#     """Decode RLE mask encoding to a binary mask."""
#     return mask_utils.decode({'counts': rle, 'size': [height, width]})


def decode_RLE_to_mask(rle_str, h, w):
    rle = {
        'counts': rle_str,
        'size': [h, w]
    }
    mask = mask_utils.decode(rle)
    return mask

# def calculate_bbox_from_mask(mask, img_height, img_width):
#     """Calculate the bounding box coordinates from a binary mask."""
#     rows = np.any(mask, axis=1)
#     cols = np.any(mask, axis=0)
#     ymin, ymax = np.where(rows)[0][[0, -1]]
#     xmin, xmax = np.where(cols)[0][[0, -1]]
#     # Normalize coordinates
#     xmin_norm = xmin / img_width
#     ymin_norm = ymin / img_height
#     width_norm = (xmax - xmin ) / img_width
#     height_norm = (ymax - ymin ) / img_height
#     return xmin_norm, ymin_norm, width_norm, height_norm


def calculate_bbox_from_mask(mask, img_height, img_width):
    if mask.max() == 0:  # No object in the mask
        return None  # or return a default box if needed
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax - xmin, ymax - ymin

def process_rle_data(input_file, output_dir):
    """Process RLE data from input file and save to output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file_path = os.path.join(output_dir, 'gt.txt')
    
    # with open(input_file, 'r') as file, open(output_file_path, 'w') as out_file:
    #     for line in file:
    #         data = line.strip().split()
    #         object_id = int(data[1]) % 1000
    #         if object_id != 0:
    #             frame_id = int(data[0])
    #             img_height = int(data[3])
    #             img_width = int(data[4])
    #             rle = data[5]
    #             class_id = int(data[2])
    #             # Decode RLE and calculate bounding box
    #             binary_mask = decode_RLE_to_mask(rle, img_height, img_width)
    #             xmin, ymin, width, height = calculate_bbox_from_mask(binary_mask, img_height, img_width)

    #             # Write the reformatted data to the output file
    #             out_file.write(f"{frame_id},{object_id}, {class_id}, {xmin},{ymin},{width},{height},{rle}\n")
    with open(input_file, 'r') as file, open(output_file_path, 'w') as out_file:
        for line in file:
            # print('line:', line)
            data = line.strip().split()
            object_id = int(data[1]) % 1000
            
            frame_id = int(data[0])
            img_height = int(data[3])
            img_width = int(data[4])
            rle = data[5]
            class_id = int(data[2])
            # Decode RLE and calculate bounding box
            binary_mask = decode_RLE_to_mask(rle, img_height, img_width)
            xmin, ymin, width, height = calculate_bbox_from_mask(binary_mask, img_height, img_width)

            # Write the reformatted data to the output file
            out_file.write(f"{frame_id},{object_id}, {class_id}, {xmin},{ymin},{width},{height},{rle}\n")



input_file_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/train/images/0020/gt/gt_mask.txt'  
output_directory = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/train/images/0020/gt'  
# process_rle_data(input_file_path, output_directory)




###################################################################################################### STEP2: CREATING INTEGRATED TXT FILES WITH CONTINUOUS IDS FROM THE GT.TXT (SUCCESSFUL) ######################################################################################################

seqs = ['0000', '0001', '0002', '0003', '0004', '0005','0006', '0007','0008', '0009', '0010','0011','0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']

label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/train/labels_with_ids/'
info_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/train/images/'


# Function to create directories if they don't exist
def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)

### Mask and boxes ###
tid_offset = 0
for seq in seqs:
    # Adapt these lines to match how you retrieve sequence information, such as image width and height
    seq_info = open(osp.join(info_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    # Load the ground truth data; adjust the path and format as needed
    gt_txt = osp.join(info_root, seq,'gt' ,'gt.txt')
    # seq_info_txt = osp.join(info_root, seq,'gt' 'gt_mask.txt')
    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    
    col_names = ['frame_id', 'track_id', 'class_id', 'x', 'y', 'w', 'h', 'rle']
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
        fid, tid, class_id, x, y, w, h, rle = row
        # if tid == 79:
        #     print(f"Track ID 79 found in {seq} at frame {fid}")
        class_id = int(class_id)
        filename_fid = int(fid)
        fid = int(fid) 
        tid = int(tid)
        tid += tid_offset
        # if not tid == tid_last:
        #     tid_curr += 1
        #     tid_last = tid
        # Calculate center x, y
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(filename_fid))
        # print('label_fpath is:', label_fpath)
        # Update the format of the label string if necessary
        label_str = '{:d} {:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {}\n'.format(
            fid, tid, class_id, x / seq_width, y / seq_height, w / seq_width, h / seq_height, rle)
        with open(label_fpath, 'a') as f:
            # print('label_fpath is:', label_fpath)
            f.write(label_str)
    tid_offset += max_tid_in_seq 


