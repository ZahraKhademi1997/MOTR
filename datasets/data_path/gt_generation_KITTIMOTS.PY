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
    
    with open(input_file, 'r') as file, open(output_file_path, 'w') as out_file:
        for line in file:
            data = line.strip().split()
            object_id = int(data[1]) % 1000
            if object_id != 0:
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


input_file_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/images/test/0000/gt/gt_mask.txt'  
output_directory = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/images/test/0000/gt'  
# process_rle_data(input_file_path, output_directory)


############################################################## TXT FILE VISUALIZATION (boxes and masks) ###############################################################
import cv2
import numpy as np
from pycocotools import mask as cocomask
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the directory containing images
images_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/MOT17/MOTS/train/images/MOTS20-02/img1/"
gt_file = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/MOT17/MOTS/train/gt_with_ids/MOTS20-02/gt.txt'
output_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/MOT17/MOTS/train/visualization/MOTS20-02"

import cv2
import numpy as np
from pycocotools import mask as cocomask
from collections import defaultdict
from pathlib import Path

def decode_RLE_to_mask(rle_str, h, w):
    rle = {
        'counts': rle_str,
        'size': [h, w]
    }
    mask = mask_utils.decode(rle)
    return mask

def visualize_frames_with_masks_and_boxes(txt_file, images_dir, output_dir):
    # Prepare output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Organize data by frames
    frame_objects = defaultdict(list)
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            object_data = parts[1:]
            frame_objects[frame_id].append(object_data)

    # Process each frame
    for frame_id, objects in frame_objects.items():
        image_path = f"{images_dir}/{frame_id:06d}.jpg"
        image = cv2.imread(image_path)
        # print(image.shape)
        image_height = image.shape[0]
        image_width = image.shape[1]
        image_shape = (image_height, image_width)
        if image is None:
            continue

        for obj in objects:
            obj_id, cx, cy, bw, bh, rle = int(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4]), obj[5]
            mask = decode_RLE_to_mask(rle, image_height, image_width)
            mask_color = (np.stack((mask,)*3, axis=-1) * np.array([0, 0, 255])).astype(np.uint8)

            image = cv2.addWeighted(image, 1, mask_color, 0.5, 0)
            print(image.shape)
            
            # x1 = int((cx - bw / 2) * image_width)
            # y1 = int((cy - bh / 2) * image_height)
            # x2 = int((cx + bw / 2) * image_width)
            # y2 = int((cy + bh / 2) * image_height)
            x1 = int((cx - bw / 2))
            y1 = int((cy - bh / 2))
            x2 = int((cx + bw / 2))
            y2 = int((cy + bh / 2))
            # print('cx:', cx, 'cy:', cy, 'bw:', bw, 'bh:', bh, 'x1:', x1, 'y1:', y1, 'x2:', x2, 'y2:', y2)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        output_image_path = f"{output_dir}/frame_{frame_id:04d}_annotated.jpg"
        cv2.imwrite(output_image_path, image)
        print(f"Processed frame {frame_id}")

    
# visualize_frames_with_masks_and_boxes(gt_file, images_dir, output_dir)

###################################################################################################### STEP2: CREATING INTEGRATED TXT FILES WITH CONTINUOUS IDS FROM THE GT.TXT (SUCCESSFUL) ######################################################################################################

seqs = ['0000', '0001', '0002', '0003', '0004', '0005','0006', '0007','0008', '0009', '0010','0011','0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
# # seqs = ['MOTS20-02']
# seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/MOT17/MOTS/train/gt_with_ids/'
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

### Just boxes ###
# tid_curr = 0
# tid_last = -1

# for seq in seqs:
#     # Adapt these lines to match how you retrieve sequence information, such as image width and height
#     # seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
#     seq_width = int(1296)
#     seq_height = int(972)

#     # Load the ground truth data; adjust the path and format as needed
#     gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
#     gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
#     idx = np.lexsort(gt.T[:2, :])
#     gt = gt[idx, :]
    
#     seq_label_root = osp.join(label_root, seq, 'img1')
#     mkdirs(seq_label_root)
    
#     # Initialize a dictionary to keep track of written object IDs for each frame
#     for fid, tid, x, y, w, h, mark, _, _, _ in gt:
#         if mark == 0:
#             continue
#         filename_fid = int(fid) - 1
#         fid = int(fid) 
#         tid = int(tid)
#         if not tid == tid_last:
#             tid_curr += 1
#             tid_last = tid
#         # Calculate center x, y
#         x += w / 2
#         y += h / 2
#         label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(filename_fid))
#         # print('label_fpath is:', label_fpath)
#         # Update the format of the label string if necessary
#         label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:d}\n'.format(
#             fid, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height, tid)
#         with open(label_fpath, 'a') as f:
#             # print('label_fpath is:', label_fpath)
#             f.write(label_str)
                      
################################################################# Creating rle format separate text file for masks only ########################################################################
# def encode_mask_to_RLE(binary_mask):
#     fortran_binary_mask = np.asfortranarray(binary_mask)
#     rle = mask_utils.encode(fortran_binary_mask)
#     return rle

# def process_mask_images(input_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for file in os.listdir(input_dir):
#         # print(input_dir)
#         if file.endswith(".png"):
#             # Extract the frame ID by removing leading zeros and the file extension
#             file_name = int(os.path.splitext(file)[0].lstrip('0') or '0')
#             frame_id = int(os.path.splitext(file)[0].lstrip('0') or '0') + 1
#             img_path = os.path.join(input_dir, file)
#             binary_mask = np.array(Image.open(img_path))
#             unique_objects = np.unique(binary_mask)[1:]  

#             rle_text_path = os.path.join(output_dir, f"{file_name:06d}.txt")
#             with open(rle_text_path, 'w') as f:
#                 for obj_id in unique_objects:
#                     object_mask = (binary_mask == obj_id)
#                     rle = encode_mask_to_RLE(object_mask)
#                     # rle = mask_utils.encode(object_mask)
#                     # print(rle)
#                     rle_str = rle['counts'].decode('ascii')
#                     # rle_str = mask_utils.decode(rle)  # Convert RLE to string format
#                     obj_id_modified = (obj_id % 1000) + 1
#                     f.write(f"{frame_id},{obj_id_modified},{rle['size'][0]},{rle['size'][1]},")
#                     f.write(f"{rle_str}\n")
#             print(f"Processed and saved RLE for {file} to {rle_text_path}")

# # Example usage
# input_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/testing/instances/0012'  
# output_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/testing/masks_with_ids/0012/'  
# process_mask_images(input_dir, output_dir)

################################################################# Creating rle format one gt text file for masks only ########################################################################
# import os
# import numpy as np
# from PIL import Image
# import pycocotools.mask as mask_utils

# def encode_mask_to_RLE(binary_mask):
#     fortran_binary_mask = np.asfortranarray(binary_mask)
#     rle = mask_utils.encode(fortran_binary_mask)
#     return rle

# def process_mask_images(input_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Open the global gt.txt file
#     rle_text_path = os.path.join(output_dir, 'gt.txt')
    
#     # Make sure to clear the file if it already exists
#     if os.path.exists(rle_text_path):
#         os.remove(rle_text_path)

#     for file in sorted(os.listdir(input_dir)):
#         if file.endswith(".png"):
#             frame_id = int(os.path.splitext(file)[0].lstrip('0') or '0') + 1
#             img_path = os.path.join(input_dir, file)
#             binary_mask = np.array(Image.open(img_path))
#             unique_objects = np.unique(binary_mask)[1:]  

#             # Open the gt.txt file in append mode
#             with open(rle_text_path, 'a') as f:
#                 for obj_id in unique_objects:
#                     object_mask = (binary_mask == obj_id)
#                     rle = encode_mask_to_RLE(object_mask)
#                     rle_str = rle['counts'].decode('ascii')
#                     obj_id_modified = (obj_id % 1000) + 1
#                     f.write(f"{frame_id},{obj_id_modified},{rle['size'][0]},{rle['size'][1]},")
#                     f.write(f"{rle_str}\n")
#             print(f"Appended RLE for {file} to {rle_text_path}")

# # Example usage
# input_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/instances/0004'
# output_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/masks_with_ids/0004/'
# process_mask_images(input_dir, output_dir)

##################################### Visualization of decoding RLE masks to Binary masks ###########################################
# import numpy as np
# import matplotlib.pyplot as plt
# from pycocotools import mask as mask_utils
# from PIL import Image

# def decode_RLE_to_mask(rle_str, h, w):
#     rle = {
#         'counts': rle_str.encode('ascii'),
#         'size': [h, w]
#     }
#     mask = mask_utils.decode(rle)
#     return mask

# def plot_masks_for_first_frame(gt_file):
#     with open(gt_file, 'r') as file:
#         lines = file.readlines()

#     # Assuming the first frame has ID 1
#     first_frame_id = 1
#     masks_for_first_frame = []

#     for line in lines:
#         frame_id, obj_id, h, w, rle_str = line.strip().split(',')[:5]
#         frame_id = int(frame_id)

#         if frame_id == first_frame_id:
#             h, w = int(h), int(w)
#             mask = decode_RLE_to_mask(rle_str, h, w)
#             masks_for_first_frame.append(mask)
#         elif frame_id > first_frame_id:
#             # Stop if we've passed the first frame
#             break

#     # Plot the masks for the first frame
#     fig, ax = plt.subplots(figsize=(10, 6))
#     combined_mask = np.max(np.stack(masks_for_first_frame), axis=0)  # Combine masks
#     ax.imshow(combined_mask, cmap='gray')
#     ax.set_title(f'Masks for Frame {first_frame_id}')
#     plt.show()



# # Example usage
# gt_file =  '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/outputs/masks_with_ids/0000/gt.txt'
# img_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/images/0005'  
# plot_masks_for_first_frame(gt_file)


