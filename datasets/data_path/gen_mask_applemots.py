################################################################# Creating rle format separate text file for masks only ########################################################################
import os
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils


def encode_mask_to_RLE(binary_mask):
    fortran_binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(fortran_binary_mask)
    return rle

def process_mask_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        # print(input_dir)
        if file.endswith(".png"):
            # Extract the frame ID by removing leading zeros and the file extension
            file_name = int(os.path.splitext(file)[0].lstrip('0') or '0')
            frame_id = int(os.path.splitext(file)[0].lstrip('0') or '0') + 1
            img_path = os.path.join(input_dir, file)
            binary_mask = np.array(Image.open(img_path))
            unique_objects = np.unique(binary_mask)[1:]  

            rle_text_path = os.path.join(output_dir, f"{file_name:06d}.txt")
            with open(rle_text_path, 'w') as f:
                for obj_id in unique_objects:
                    object_mask = (binary_mask == obj_id)
                    rle = encode_mask_to_RLE(object_mask)
                    # rle = mask_utils.encode(object_mask)
                    # print(rle)
                    rle_str = rle['counts'].decode('ascii')
                    # rle_str = mask_utils.decode(rle)  # Convert RLE to string format
                    obj_id_modified = (obj_id % 1000) + 1
                    f.write(f"{frame_id},{obj_id_modified},{rle['size'][0]},{rle['size'][1]},")
                    f.write(f"{rle_str}\n")
            print(f"Processed and saved RLE for {file} to {rle_text_path}")


# input_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/instances/0000'  
# output_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/masks_with_ids/0000/'  
# process_mask_images(input_dir, output_dir)

################################################################# Creating rle format one gt text file for masks only ########################################################################
import os
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils

def encode_mask_to_RLE(binary_mask):
    fortran_binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(fortran_binary_mask)
    return rle

def process_mask_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the global gt.txt file
    rle_text_path = os.path.join(output_dir, 'gt.txt')
    
    # Make sure to clear the file if it already exists
    if os.path.exists(rle_text_path):
        os.remove(rle_text_path)

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".png"):
            frame_id = int(os.path.splitext(file)[0].lstrip('0') or '0') + 1
            img_path = os.path.join(input_dir, file)
            binary_mask = np.array(Image.open(img_path))
            unique_objects = np.unique(binary_mask)[1:]  

            # Open the gt.txt file in append mode
            with open(rle_text_path, 'a') as f:
                for obj_id in unique_objects:
                    object_mask = (binary_mask == obj_id)
                    rle = encode_mask_to_RLE(object_mask)
                    rle_str = rle['counts'].decode('ascii')
                    obj_id_modified = (obj_id % 1000) + 1
                    f.write(f"{frame_id},{obj_id_modified},{rle['size'][0]},{rle['size'][1]},")
                    f.write(f"{rle_str}\n")
            print(f"Appended RLE for {file} to {rle_text_path}")


# input_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/instances/0004'
# output_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/masks_with_ids/0004/'
# process_mask_images(input_dir, output_dir)

##################################### Visualization of decoding RLE masks to Binary masks ###########################################
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from PIL import Image

def decode_RLE_to_mask(rle_str, h, w):
    rle = {
        'counts': rle_str.encode('ascii'),
        'size': [h, w]
    }
    mask = mask_utils.decode(rle)
    return mask

def plot_masks_for_first_frame(gt_file):
    with open(gt_file, 'r') as file:
        lines = file.readlines()

    # Assuming the first frame has ID 1
    first_frame_id = 1
    masks_for_first_frame = []

    for line in lines:
        frame_id, obj_id, h, w, rle_str = line.strip().split(',')[:5]
        frame_id = int(frame_id)

        if frame_id == first_frame_id:
            h, w = int(h), int(w)
            mask = decode_RLE_to_mask(rle_str, h, w)
            masks_for_first_frame.append(mask)
        elif frame_id > first_frame_id:
            # Stop if we've passed the first frame
            break

    # Plot the masks for the first frame
    fig, ax = plt.subplots(figsize=(10, 6))
    combined_mask = np.max(np.stack(masks_for_first_frame), axis=0)  # Combine masks
    ax.imshow(combined_mask, cmap='gray')
    ax.set_title(f'Masks for Frame {first_frame_id}')
    plt.show()



# gt_file =  '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/outputs/masks_with_ids/0000/gt.txt'
# img_dir = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/APPLE_MOTS/train/images/0005'  
# plot_masks_for_first_frame(gt_file)