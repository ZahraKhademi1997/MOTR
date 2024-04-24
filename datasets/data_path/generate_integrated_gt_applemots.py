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
def encode_mask_to_RLE(binary_mask):
    fortran_binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(fortran_binary_mask)
    return rle['counts'].decode('ascii') 

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
                            rle = encode_mask_to_RLE(obj_mask)
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
                            f.write(f'{frame_id},{object_id},{xmin},{ymin},{w},{h}, 1, {rle}\n')


                    if np.max(obj_ids) > track:
                        track = np.max(obj_ids)
                else:
                    continue
            track_id = track % 1000
            ls = [frames, track_id, mask]
            dicts[dataset] = ls

    df = pd.DataFrame.from_dict(dicts, orient='index', columns=['#frames', '#tracks', '#masks'])
    return df



# train_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/APPLE_MOTS/testing/"
# bbox_save_dir= "/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/APPLE_MOTS/train/test_gt_files/"

# overview(train_dir,bbox_save_dir)




############################################################## TXT FILE VISUALIZATION ###############################################################
# Path to the directory containing images
# images_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/images/0005"
# gt_file = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt/gt_after_remapping/0005_gt.txt"
# output_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/test_gt/vis/image5"

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
        
        
# plot_bboxes_and_ids(gt_file, images_dir, output_dir)

###################################################################################################### STEP2: CREATING INTEGRATED TXT FILES WITH CONTINUOUS IDS FROM THE GT.TXT (SUCCESSFUL) ######################################################################################################

# seqs = ['0006', '0007', '0008', '0010', '0011', '0012']
# # seqs = ['0000', '0001', '0002', '0003', '0004', '0005']
# seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/APPLE_MOTS/testing/images/'
# label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/data/Dataset/APPLE_MOTS/testing/labels_with_ids_prototype_test/'


# # Function to create directories if they don't exist
# def mkdirs(path):
#     if not osp.exists(path):
#         os.makedirs(path)

### Mask and boxes ###
tid_offset = 0
for seq in seqs:
    # Adapt these lines to match how you retrieve sequence information, such as image width and height
    # seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(1296)
    seq_height = int(972)

    # Load the ground truth data; adjust the path and format as needed
    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    
    col_names = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'mark', 'rle']
    # col_types = {'frame_id': int, 'track_id': int, 'x': float, 'y': float, 'w': float, 'h': float, 'mark': int, 'rle': str}

    # gt = pd.read_csv(gt_txt, header=None, names=col_names, dtype=col_types)
    gt = pd.read_csv(gt_txt, header=None, names=col_names)
    gt.sort_values(by=['frame_id', 'track_id'], inplace=True)
    max_tid_in_seq = gt['track_id'].max() if not gt.empty else 0
    print('max_tid_in_seq:', max_tid_in_seq)

    for index, row in gt.iterrows():
        fid, tid, x, y, w, h, mark, rle = row
        
        if mark == 0:
            continue
        filename_fid = int(fid) - 1
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
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {}\n'.format(
            fid, tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, rle)
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



