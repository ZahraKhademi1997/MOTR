from pycocotools.coco import COCO
import os
import json
import requests
from io import BytesIO
from PIL import Image
import pycocotools.mask as mask_utils
from pycocotools.mask import encode, decode
import numpy as np
import cv2
import os.path as osp
import pandas as pd 
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as mask_utils
import ast 

# Downloading and converting json file to person class only
def filter_person_category(data_dir, dataset_type):
    ann_file = os.path.join(data_dir, f'instances_{dataset_type}.json')
    coco = COCO(ann_file)
    
    # Category IDs for the person category
    catIds = coco.getCatIds(catNms=['person'])
    # Get all images containing the 'person' category
    imgIds = coco.getImgIds(catIds=catIds)
    
    # Load and filter annotations
    person_imgs = coco.loadImgs(imgIds)
    person_anns = coco.loadAnns(coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None))

    # Save filtered annotations into a new file
    with open(f'{data_dir}/person_instances_{dataset_type}.json', 'w') as f:
        json.dump({
            "images": person_imgs,
            "annotations": person_anns,
            "categories": coco.loadCats(catIds)
        }, f)
        
    
    # Create a directory to store downloaded images
    # img_dir = os.path.join(data_dir, 'images', dataset_type)
    # os.makedirs(img_dir, exist_ok=True)

    # # Download images
    # for img_info in person_imgs:
    #     img_url = img_info['coco_url']
    #     img_path = os.path.join(img_dir, img_info['file_name'])
    #     if not os.path.exists(img_path):  # Avoid downloading if already exists
    #         response = requests.get(img_url)
    #         image = Image.open(BytesIO(response.content))
    #         image.save(img_path)
    #         print(f'Downloaded {img_path}')

# usage
# data_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/'
# filter_person_category(data_dir, 'train2014')
# filter_person_category(data_dir, 'val2017')


# Converting json file to MOT format text file
def calculate_bbox_from_mask(mask, img_height, img_width):
    if mask.max() == 0:  # No object in the mask
        return None  # or return a default box if needed
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax - xmin, ymax - ymin

# def decode_RLE_to_mask(rle, h, w):
#     rle = {
#         'counts': rle,
#         'size': [h, w]
#     }
#     return mask_utils.decode(rle)

def decode_RLE_to_mask(rle, h, w):
    return mask_utils.decode(rle)

def json_to_mots(input_json_path, output_txt_path):
    # Load JSON file
    with open(input_json_path, 'r') as file:
        data = json.load(file)
        # print('data:', data)
    output_dir = os.path.dirname(output_txt_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Prepare to find image dimensions
    images = {img['id']: img for img in data['images']}
    
    with open(output_txt_path, 'w') as file:
        for annotation in data['annotations']:
            frame_id = annotation['image_id']
            object_id = annotation['id']
            class_id = annotation['category_id']
            xmin, ymin, width, height = annotation['bbox']
            # print('bbox',annotation['bbox'])
            
            img_details = images[frame_id]
            img_width, img_height = img_details['width'], img_details['height']
            
            # Normalize bbox coordinates
            # norm_xmin = xmin / img_width
            # norm_ymin = ymin / img_height
            # norm_width = width / img_width
            # norm_height = height / img_height
            
            # Check the segmentation type and convert to binary mask
            segmentation = annotation['segmentation']
            
            if isinstance(segmentation, list):  # Handling polygon cases
                for seg in segmentation:
                    # Convert each float coordinate pair to an integer
                    int_coords = [int(coord) for coord in seg]
                    rle = mask_utils.frPyObjects([int_coords], img_height, img_width)
                    binary_mask = mask_utils.decode(rle)
                    rle_encoded = mask_utils.encode(np.asfortranarray(binary_mask))
                    # print('rle:', rle_encoded)
                    rle_str = rle_encoded[0]['counts'].decode('utf-8') if isinstance(rle_encoded[0]['counts'], bytes) else rle_encoded[0]['counts']
                    seq_height, seq_width = rle_encoded[0]['size']

                    # Write to output file
                    # file.write(f"{frame_id},{object_id},{norm_xmin},{norm_ymin},{norm_width},{norm_height},{rle_str}\n")
                    file.write(f"{frame_id},{object_id},{xmin},{ymin},{width},{height},{seq_width}, {seq_height}, {rle_str}\n")

# json_to_mots("/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/person_instances_train2014.json", "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/images/COCO_00/gt.txt")


# Creating distinct text files
# seqs = ['COCO_00']

# seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/images/'
# label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/labels_with_ids/'


# # Function to create directories if they don't exist
# def mkdirs(path):
#     if not osp.exists(path):
#         os.makedirs(path)

# ## Mask and boxes ###
# tid_curr = 0
# tid_last = -1
# tid_offset = 0
# for seq in seqs:
#     # Load the ground truth data; adjust the path and format as needed
#     gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
#     seq_label_root = osp.join(label_root, seq, 'img1')
#     mkdirs(seq_label_root)
    
#     col_names = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'seq_width', 'seq_height', 'rle']
#     col_types = {'frame_id': int, 'track_id': int, 'x': float, 'y': float, 'w': float, 'h': float, 'seq_width': int, 'seq_height': int, 'rle': str}

#     gt = pd.read_csv(gt_txt, header=None, names=col_names, dtype=col_types)
#     gt = pd.read_csv(gt_txt, header=None, names=col_names)
#     gt.sort_values(by=['frame_id', 'track_id'], inplace=True)
#     max_tid_in_seq = gt['track_id'].max() if not gt.empty else 0
#     print('max_tid_in_seq:', max_tid_in_seq)
    
   
#     for index, row in gt.iterrows():
#         fid, tid, x, y, w, h, seq_width, seq_height, rle = row
        
#         filename_fid = 'COCO_train2014_{:012d}'.format(int(fid))
#         fid = int(fid) 
#         tid = int(tid)
#         tid += tid_offset
#         # if not tid == tid_last:
#         #     tid_curr += 1
#         #     tid_last = tid
#         # Calculate center x, y
#         x += w / 2
#         y += h / 2
#         label_fpath = osp.join(seq_label_root, f'{filename_fid}.txt')
#         # print('label_fpath is:', label_fpath)
#         # Update the format of the label string if necessary
#         label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {}\n'.format(
#             fid, tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height , rle)
#         with open(label_fpath, 'a') as f:
#             # print('label_fpath is:', label_fpath)
#             f.write(label_str)
#     tid_offset += max_tid_in_seq 


# # Comparing txt and jpg
# # Define the paths to your directories
# txt_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/labels_with_ids/COCO_00/img1'
# jpg_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/images/COCO_00/img1'

# # Collect all text file basenames (without extensions)
# txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_dir) if f.endswith('.txt')}

# # Loop through all JPEG files in the jpg directory
# for jpg_file in os.listdir(jpg_dir):
#     if jpg_file.endswith('.jpg'):
#         # Check if there's a corresponding text file (without the .jpg extension)
#         jpg_basename = os.path.splitext(jpg_file)[0]
#         if jpg_basename not in txt_files:
#             # If no corresponding text file, delete the JPEG file
#             jpg_path = os.path.join(jpg_dir, jpg_file)
#             os.remove(jpg_path)
#             print(f"Deleted: {jpg_path}")
            
# # Collect all JPEG file basenames (without extensions)
# # jpg_files = {os.path.splitext(f)[0] for f in os.listdir(jpg_dir) if f.endswith('.jpg')}

# # # Loop through all text files in the txt directory
# # for txt_file in os.listdir(txt_dir):
# #     if txt_file.endswith('.txt'):
# #         # Check if there's a corresponding JPEG file (without the .txt extension)
# #         txt_basename = os.path.splitext(txt_file)[0]
# #         if txt_basename not in jpg_files:
# #             # If no corresponding JPEG file, delete the text file
# #             txt_path = os.path.join(txt_dir, txt_file)
# #             os.remove(txt_path)
# #             print(f"Deleted: {txt_path}")


# Plotting 
# Path to the image and txt file
# image_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/images/COCO_00/img1/COCO_train2014_000000000109.jpg'
# txt_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask_DAB_DN_Track/data/Dataset/COCO/train/labels_with_ids/COCO_00/img1/COCO_train2014_000000000109.txt'

# # Check if the image file exists
# if not os.path.exists(image_path):
#     print(f"Error: Image file does not exist at {image_path}")
# else:
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Failed to load image from {image_path}")
#     else:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Function to draw a rectangle on the image
#         def draw_bbox(img, bbox, h, w):
#             print(bbox)
#             np_img = np.array(img)
#             fig, ax = plt.subplots(1, figsize=(12, 8))
#             ax.imshow(np_img)
#             cx, cy, bw, bh = bbox
#             x1 = (cx - bw / 2) * w
#             y1 = (cy - bh / 2) * h
#             x2 = (cx + bw / 2) * w
#             y2 = (cy + bh / 2) * h
#             rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)

#         # Function to apply mask on the image
#         def apply_mask(img, mask):
#             for c in range(3):
#                 img[:, :, c] = np.where(mask == 1, 255, img[:, :, c])
#             return img

#         # Read the txt file and process each entry
#         with open(txt_path, 'r') as file:
#             for line in file:
#                 parts = line.strip().split()
#                 fid, tid, x, y, w, h, rle_str = parts
#                 # print('parts:', parts)
#                 bbox = [float(x), float(y), float(w), float(h)]
#                 print(bbox)
#                 draw_bbox(image, bbox,image.shape[0], image.shape[1]) # height, width
#                 # rle_bytes = bytes(rle_str, 'utf-8')  # Convert string to bytes, handling escapes correctly
#                 # print(rle_bytes)

#                 rle = {
#                     'counts': rle_str, # Ensure it's a byte string
#                     'size': [image.shape[0], image.shape[1]]  # Height, width of the image
#                 }
#                 # print('rle:', rle)
#                 try:
#                     # Decode RLE to binary mask
#                     mask = mask_utils.decode(rle)
#                     print("Mask decoded successfully.")
#                 except ValueError as e:
#                     print(f"Failed to decode RLE: {e}")

#                 image = apply_mask(image, mask)

#         # Display the image
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         plt.axis('off')
#         plt.show()