###################################################################################################### STEP1: CREATING TXT FILES WITH CONTINUOUS IDS FROM THE GT.TXT (SUCCESSFUL) ######################################################################################################
import os
import os.path as osp
import numpy as np


seqs = ['0000', '0001', '0002', '0003', '0004', '0005']

seq_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/images/'
label_root = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/labels_with_ids/'



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
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            fid, tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            # print('label_fpath is:', label_fpath)
            f.write(label_str)
###################################################################################################################################################################################################################################################################

#################################################################### STEP2: Visualization to confirm data processing #####################################################################

def save_img_with_annotations(image, bboxes, masks, obj_ids, out_dir, seq_dir, file_name):
    """
    Saves an image with bounding boxes and masks annotations.
    
    Parameters:
    - image: Image array.
    - bboxes: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    - masks: List of binary mask arrays corresponding to object IDs.
    - obj_ids: List of object IDs corresponding to bounding boxes and masks.
    - out_dir: Directory to save the annotated image.
    - file_name: Name of the file to be saved.
    """
    
    seq_out_dir = os.path.join(out_dir, seq_dir)
    os.makedirs(seq_out_dir, exist_ok=True)
    
    
    fig, ax = plt.subplots(1, figsize=(14, 14))
    ax.imshow(image)

    # Add annotations for each object
    for bbox, mask, obj_id in zip(bboxes, masks, obj_ids):
        x_min, y_min, x_max, y_max = bbox
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 2, f'ID: {obj_id}', color='white', fontsize=5,
                bbox=dict(facecolor='red', alpha=0.5))

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Assuming mask is a boolean array here, directly use it for overlay
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='cool', alpha=0.5)

    plt.axis('off')
    full_out_path = os.path.join(seq_out_dir, file_name)
    plt.savefig(full_out_path, bbox_inches='tight')
    plt.close()
    
    
def process_and_save_all_images(image_base_dir, label_base_dir, mask_base_dir, out_dir):
    seq_dirs = ['0000', '0001', '0002', '0003', '0004', '0005']
    for seq_dir in seq_dirs:
        image_dir = os.path.join(image_base_dir, seq_dir)
        label_dir = os.path.join(label_base_dir, seq_dir)
        mask_dir = os.path.join(mask_base_dir, seq_dir)

        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        # print(image_files)
        label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt') for x in image_files]
        mask_files = [x.replace('images', 'instances').replace('.png', '.png') for x in image_files]
        
        for img_file, label_file, mask_file in zip(image_files, label_files, mask_files):
            image = Image.open(img_file)
            mask = Image.open(mask_file)
            mask_np = np.array(mask)
            w, h = image.size
            assert w > 0 and h > 0, "Invalid image {} with shape {} {}".format(img_file, w, h)
            obj_idx_offset = 0
            targets = {}
            targets['dataset'] = 'AppleMOTS'
            targets['boxes'] = []
            targets['masks'] = []
            targets['area'] = []
            targets['iscrowd'] = []
            targets['labels'] = []
            targets['obj_ids'] = []
            targets['image_id'] = torch.as_tensor([int(os.path.splitext(os.path.basename(img_file))[0])])
            targets['size'] = torch.as_tensor([h, w])
            targets['orig_size'] = torch.as_tensor([h, w])

            if osp.isfile(label_file):
                labels0 = np.loadtxt(label_file, dtype=np.float32).reshape(-1, 6)

                labels = labels0.copy()
                labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
                labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
                labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
                labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
            else:
                raise ValueError('Invalid label path: {}'.format(label_file))
            
            filtered_boxes = []
            filtered_masks = []
            filtered_obj_ids = []
            filtered_areas = []
            filtered_iscrowd = []
            filtered_labels = []

            for label in labels:
                x_min, y_min, x_max, y_max = label[2:6]
                cropped_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                if np.any(cropped_mask):
                    new_mask = np.zeros_like(mask_np)
                    new_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = (cropped_mask > 0).astype(np.uint8)
                    
                    filtered_boxes.append(label[2:6].tolist())
                    filtered_masks.append(new_mask)
                    filtered_areas.append((x_max - x_min) * (y_max - y_min))
                    filtered_iscrowd.append(0)  # Assuming single class, modify as necessary
                    filtered_labels.append(0)  # Assuming single class, modify as necessary
                    obj_id = label[1] + obj_idx_offset  # Assuming obj_idx_offset is defined
                    filtered_obj_ids.append(obj_id)

            # Update targets with filtered and adjusted information
            if filtered_boxes:  # Ensure there's at least one valid mask-box pair
                targets['boxes'] = torch.as_tensor(filtered_boxes, dtype=torch.float32).reshape(-1, 4)
                targets['masks'] = torch.as_tensor(np.array(filtered_masks), dtype=torch.uint8)
                targets['area'] = torch.as_tensor(filtered_areas)
                targets['iscrowd'] = torch.as_tensor(filtered_iscrowd)
                targets['labels'] = torch.as_tensor(filtered_labels)
                targets['obj_ids'] = torch.as_tensor(filtered_obj_ids)
            else:
                print("No valid mask-box pairs found.")
            
            # Save annotated image
            file_name = os.path.basename(img_file)
            # save_img_with_annotations(image, targets['boxes'], targets['masks'], targets['obj_ids'], out_dir, seq_dir, file_name)


# image_base_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/images"
# label_base_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/labels_with_ids"
# mask_base_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/data/Dataset/APPLE_MOTS/train/instances"
# out_dir = "/home/zahra/Documents/Projects/prototype/MOTR-codes/test/MOTR-main/outputs/test3"
# process_and_save_all_images(image_base_dir, label_base_dir, mask_base_dir, out_dir)
    
    
