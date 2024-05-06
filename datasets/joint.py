

"""
MOT dataset which returns image_id for evaluation.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
import pycocotools.mask as rletools
import glob
from pycocotools import mask as mask_util
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime
import random
import torchvision.transforms.functional as F
import numpy as np
from pycocotools import mask as mask_utils

class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.video_dict = {}
        
        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        # self.mask_files = [(x.replace('images', 'masks_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
        #                     for x in self.img_files]
        # The number of images per sample: 1 + (num_frames - 1) * interval.
        # The number of valid samples: num_images - num_image_per_sample + 1.
        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval

        self._register_videos()

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0

    def _register_videos(self):
        for label_name in self.label_files:
            video_name = '/'.join(label_name.split('/')[:-1]) +'/img1'
            if video_name not in self.video_dict:
                # print("register {}-th video: {} ".format(len(self.video_dict) + 1, video_name))
                self.video_dict[video_name] = len(self.video_dict)
                # assert len(self.video_dict) <= 300

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        ##############################################################################################
        # (4)
        gt_instances.masks = targets['masks']
        ##############################################################################################
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances
    
    
    # Modifying function to load masks and bboxes
    def _pre_single_frame(self, idx: int):
        
        # Converting RLE to binary mask
        def decode_RLE_to_mask(rle_str, h, w):
            rle = {
                'counts': rle_str,
                'size': [h, w]
            }
            mask = mask_utils.decode(rle)
            return mask


        # Visualization of the gt attributes
        def plot_frame_with_annotations(img, targets):
            """
            Plot the image with bounding boxes, masks, and object IDs from the targets.

            :param img: PIL Image object of the frame.
            :param targets: Dictionary containing 'boxes', 'masks', 'obj_ids' etc.
            """
            # Convert image to numpy array
            np_img = np.array(img)

            # Create a figure and axis for plotting
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(np_img)

            boxes = targets['boxes'].cpu().numpy()  # Assuming targets['boxes'] is a tensor
            masks = targets['masks']  # Assuming targets['masks'] is a list of tensors
            obj_ids = targets['obj_ids'].cpu().numpy()  # Assuming targets['obj_ids'] is a tensor

            for box, mask, obj_id in zip(boxes, masks, obj_ids):
                # Draw the bounding box
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

                # Label the bounding box with the object ID
                ax.text(box[0], box[1], f'ID: {obj_id}', bbox=dict(facecolor='blue', alpha=0.5), clip_on=True, color='white')

                # Overlay the mask
                mask_np = mask.cpu().numpy()
                ax.imshow(np.ma.masked_where(mask_np == 0, mask_np), alpha=0.5, cmap='cool')

            plt.show()
    
        
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        # mask_path = self.mask_files[idx]  

        img = Image.open(img_path)
        w, h = img._size
        targets = {}
        
        video_name = '/'.join(label_path.split('/')[:-1])+ '/img1'
        obj_idx_offset = self.video_dict[video_name] * 1000000  # 1000000 unique ids is enough for a video.
        
        if 'crowdhuman' in img_path:
            targets['dataset'] = 'CrowdHuman'
        elif 'MOT17' in img_path:
            targets['dataset'] = 'MOT17'
        elif 'APPLE_MOTS' in img_path:
            targets['dataset'] = 'APPLE_MOTS'
        else:
            raise NotImplementedError()
        
        targets['boxes'] = []
        targets['masks'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])

        # label_data = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)

        # Process each object in label file
        if osp.isfile(label_path):
            # Load combined data (bbox + RLE mask)
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    frame_id = parts[0] 
                    object_id= int(parts[1]) 
                    normalized_bbox = list(map(float, parts[2:6])) 
                    rle_str = parts[6]
                    cx, cy, bw, bh = normalized_bbox
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h


                    # Decode RLE to mask
                    mask = decode_RLE_to_mask(rle_str, int(h), int(w))

                    # Append data to targets
                    targets['boxes'].append([x1, y1, x2, y2])
                    targets['masks'].append(mask)
                    targets['area'].append((x2 - x1) * (y2 - y1))
                    targets['labels'].append(0)  
                    targets['iscrowd'].append(0)
                    obj_id = object_id + obj_idx_offset if object_id >= 0 else object_id
                    targets['obj_ids'].append(obj_id)
                    

        # Convert lists to tensors
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['masks'] = [torch.from_numpy(mask) for mask in targets['masks']]
        targets['masks'] = torch.stack(targets['masks'])
        # print(targets)
        # plot_frame_with_annotations(img, targets)
        return img, targets

        
    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            
            # Check and align the counts of masks and boxes
            # num_masks = len(targets_i['masks'])
            # num_boxes = len(targets_i['boxes'])
            # print('num_masks are:', num_masks)
            # print('num_boxes are:', num_boxes)
            # if num_masks != num_boxes:
            #     # Optionally log the mismatch
            #     print(f'Mismatch in frame {i}: num_masks={num_masks}, num_boxes={num_boxes}')
            
            images.append(img_i)
            targets.append(targets_i)
            # print('targets in pre_continuous_frames is:', targets)
    
        return images, targets
    

    def __getitem__(self, idx):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval)
        # Check targets before transformation
        
        data = {}
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]
        # for t in targets:
        #     print(f"Before transform: masks={len(t['masks'])}, boxes={len(t['boxes'])}")
            
        if transform is not None:
            images, targets = transform(images, targets)
            
            # Check targets after transformation
            # for t in targets:
            #     print(f"After transform: masks={len(t['masks'])}, boxes={len(t['boxes'])}")
        gt_instances = []
        
        def plot_image_with_bboxes_and_masks(img, bboxes, masks, title):
            """
            Plots an image with bounding boxes and masks.
            img: PIL image, numpy array, or PyTorch tensor
            bboxes: List of bounding boxes, each bbox is [x_min, y_min, x_max, y_max]
            masks: List of masks, each mask is a 2D numpy array
            title: Title for the plot
            """
            # Convert PyTorch tensor to numpy array and change shape to (H, W, C)
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()

            # Convert to numpy array if img is a PIL image
            if isinstance(img, Image.Image):
                img = np.array(img)

            # Ensure img is in the correct format (H, W, C) for RGB images
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # If channels are first
                img = img.transpose(1, 2, 0)

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            # print('imagse shape is:', img.shape)
  

            for bbox, mask in zip(bboxes, masks):
                # print('bbox are:', bbox)
                x_center, y_center, bbox_width, bbox_height = bbox
                x_min = (x_center - bbox_width / 2) * img_width
                y_min = (y_center - bbox_height / 2) * img_height
                x_max = (x_center + bbox_width / 2) * img_width
                y_max = (y_center + bbox_height / 2) * img_height

                # Draw the bounding box as a rectangle
                rect = patches.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min, 
                    y_max - y_min, 
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                
                ### For saving ###
                ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='spring', alpha=0.5)

            plt.title(title)
            file_name = title.replace('', '_') + '.png'
            # Save the figure
            output_path = os.path.join('/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask_AppleMOTS-InstanceAware/output/plot_check_joint', file_name)
            plt.savefig(output_path)
            plt.close(fig)  # Close the figure to free memory
                
                ### For plotting ###
                # Overlay the mask
            #     ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='spring', alpha=0.5)

            # plt.title(title)
            # plt.show()
        
        
        filtered_gt_instances = []
        filtered_images = []
        for img_i, targets_i in zip(images, targets):
            # print('target-i is:', targets_i)
            #############################################################################
            # () 
            # print('target_i before is:', targets_i)
            # if len(targets_i['masks']) != len(targets_i['boxes']):
            #     lengths = {field: len(targets_i[field]) for field in ['boxes', 'labels', 'obj_ids', 'area', 'iscrowd', 'masks']}
            #     print(f"Field lengths in frame {idx}: {lengths}")
                # plot_image_with_bboxes_and_masks(img_i, targets_i['boxes'], targets_i['masks'], f"Frame {idx}")

            if len(targets_i['masks']) != len(targets_i['boxes']):
                print(f'Before equalizing - Frame: Masks={len(targets_i["masks"])}, Boxes={len(targets_i["boxes"])}, Labels = {len(targets_i["labels"])}')
                # plot_image_with_bboxes_and_masks(img_i, targets_i['boxes'], targets_i['masks'], f"Before-Frame {idx}")

                # Display the (xmin,ymin) and (xmax,ymax)
                # fig, ax = plt.subplots(figsize=(10, 8))
                # image_np = img_i.permute(1, 2, 0).cpu().numpy()
                # # image_np = (image_np * 255).astype('uint8')
                # ax.imshow(image_np)
                
                # for bbox in targets_i['boxes']:
                #     x_center, y_center, bbox_width, bbox_height = bbox
                #     x_min = max(int((x_center - bbox_width / 2) * img_width), 0)
                #     y_min = max(int((y_center - bbox_height / 2) * img_height), 0)
                #     x_max = min(int((x_center + bbox_width / 2) * img_width), img_width - 1)
                #     y_max = min(int((y_center + bbox_height / 2) * img_height), img_height - 1)

                #     # Debug output
                #     print(f'BBox corners: (x_min, y_min) = ({x_min}, {y_min}), (x_max, y_max) = ({x_max}, {y_max})')

                #     # Plot the min and max points
                #     ax.plot(x_min, y_min, 'ro')  # red point for (x_min, y_min)
                #     ax.plot(x_max, y_max, 'bs')  # blue square for (x_max, y_max)

                # plt.show()

                
                def create_bbox_from_mask(mask):
                    pos = np.where(mask)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    bbox = [xmin, ymin, xmax, ymax]
                    return bbox
                
                # Initialize new lists for valid boxes and associated data
                new_boxes = []
                new_labels = []
                new_obj_ids = []
                new_area = []
                new_iscrowd = []
                new_masks = []
                
                # img_width, img_height = targets_i['size'][1], targets_i['size'][0]
                
                # # First, create bboxes for masks without any
                # mask_to_box_mapping = [False] * len(targets_i['masks'])  # Keep track of which masks have boxes
                        
                # for box, label, obj_id, area, iscrowd, mask in zip(targets_i['boxes'], targets_i['labels'], targets_i['obj_ids'], targets_i['area'], targets_i['iscrowd'], targets_i['masks']):
                #     # Get the bounding box coordinates
                #     x_center, y_center, bbox_width, bbox_height = box
                #     x_min = max(int((x_center - bbox_width / 2) * img_width), 0)
                #     y_min = max(int((y_center - bbox_height / 2) * img_height), 0)
                #     x_max = min(int((x_center + bbox_width / 2) * img_width), img_width - 1)
                #     y_max = min(int((y_center + bbox_height / 2) * img_height), img_height - 1)

                #     # Get the mask as a boolean tensor
                #     mask_bool = mask.bool() if isinstance(mask, torch.Tensor) else mask.astype(bool)
                    
                #     # Find the central point of the mask
                #     y_indices, x_indices = torch.where(mask_bool)
                #     if len(y_indices) > 0 and len(x_indices) > 0:
                #         mask_center_x = torch.mean(x_indices.float()).int() 
                #         mask_center_y = torch.mean(y_indices.float()).int()
                        
                #         # Check if the central point of the mask is within the bounding box
                #         if (x_min <= mask_center_x <= x_max) and (y_min <= mask_center_y <= y_max):
                #             new_boxes.append([x_center, y_center, bbox_width, bbox_height])
                #             new_labels.append(label)
                #             new_obj_ids.append(obj_id)
                #             new_area.append(area)
                #             new_iscrowd.append(iscrowd)
                #             new_masks.append(mask)
                
                img_width, img_height = targets_i['size'][1], targets_i['size'][0]

                # First, create bboxes for masks without any
                mask_to_box_mapping = [False] * len(targets_i['masks'])  # Keep track of which masks have boxes

                for i, mask in enumerate(targets_i['masks']):
                    
                    # Convert the mask to a NumPy array if it's a tensor
                    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

                    # Initialize an empty bbox_array as a NumPy array with the same shape as the mask
                    bbox_array = np.zeros(mask_np.shape, dtype=bool)

                    # Check if any existing bbox corresponds to this mask
                    for box in targets_i['boxes']:
                        # Convert box center and size to pixel coordinates
                        x_center, y_center, bbox_width, bbox_height = box
                        x_min = max(int((x_center - bbox_width / 2) * img_width), 0)
                        y_min = max(int((y_center - bbox_height / 2) * img_height), 0)
                        x_max = min(int((x_center + bbox_width / 2) * img_width), img_width - 1)
                        y_max = min(int((y_center + bbox_height / 2) * img_height), img_height - 1)
                        
                        # Create a boolean array for the bbox
                        bbox_array[y_min:y_max+1, x_min:x_max+1] = True
                        
                        # Check if the mask overlaps with the bbox
                        if np.any(mask_np & bbox_array):
                            mask_to_box_mapping[i] = True
                            continue
                    
                    # If no bbox corresponds to this mask, create a new one
                    if not mask_to_box_mapping[i]:
                        new_bbox = create_bbox_from_mask(mask.numpy() if isinstance(mask, torch.Tensor) else mask)
                        new_boxes.append(new_bbox)
                        new_labels.append(1)  # Placeholder: The actual label for this mask
                        new_obj_ids.append(-1)  # Placeholder: The actual object ID for this mask
                        new_area.append((new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]))
                        new_iscrowd.append(0)  # Placeholder: The actual crowd status for this mask
                        new_masks.append(mask)

                # Now, filter out bboxes that don't have corresponding masks
                for i, (box, label, obj_id, area, iscrowd, mask) in enumerate(zip(targets_i['boxes'], targets_i['labels'], targets_i['obj_ids'], targets_i['area'], targets_i['iscrowd'], targets_i['masks'])):
                    if mask_to_box_mapping[i]:
                        new_boxes.append(box.tolist())
                        new_labels.append(label)
                        new_obj_ids.append(obj_id)
                        new_area.append(area)
                        new_iscrowd.append(iscrowd)
                        new_masks.append(mask)
                 

                
                # Update targets_i with filtered data
                targets_i['boxes'] = torch.as_tensor(new_boxes, dtype=torch.float32)
                targets_i['labels'] = torch.as_tensor(new_labels, dtype=torch.int64)
                targets_i['obj_ids'] = torch.as_tensor(new_obj_ids, dtype=torch.int64)
                targets_i['area'] = torch.as_tensor(new_area, dtype=torch.float32)
                targets_i['iscrowd'] = torch.as_tensor(new_iscrowd, dtype=torch.int64)
                masks_np = np.stack([mask.numpy() for mask in new_masks])
                targets_i['masks'] = torch.as_tensor(masks_np, dtype=torch.uint8)

                # print('target[bboxes] are:', targets_i['boxes'])
                # plot_image_with_bboxes_and_masks(img_i, targets_i['boxes'], targets_i['masks'], f"After-Frame {idx}")
                print(f'After equalizing - Frame: Masks={len(targets_i["masks"])}, Boxes={len(targets_i["boxes"])}') 
            
            # Checkpoint to verify the lengths after filtering
            if len(targets_i['masks']) != len(targets_i['boxes']):
                lengths = {field: len(targets_i[field]) for field in ['boxes', 'labels', 'obj_ids', 'area', 'iscrowd', 'masks']}
                print(f"Field lengths in frame {idx}: {lengths}")
                # plot_image_with_bboxes_and_masks(img_i, targets_i['boxes'], targets_i['masks'], f"Frame {idx}")
                
                
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            
            
        #     if len(targets_i['boxes']) > 0:  # or any other condition you use to validate the data
        #         # If valid instances are found, convert the filtered targets into Instances format
        #         gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])

        #         # Append the valid data to the filtered lists
        #         filtered_images.append(img_i)
        #         filtered_gt_instances.append(gt_instances_i)
        #     else:
        #         # If no valid instances are found in this frame, skip to the next frame
        #         continue
            
        # data.update({
        #     'imgs': filtered_images,
        #     'gt_instances': filtered_gt_instances,
        # })
        
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
        })
        
        #############################################################################
    
        if self.args.vis:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        return self.item_num


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)



def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


###############################################################################
# () Adding data augmentation for AppleMOTS
# def make_transforms_for_applemots(image_set, args=None):

#     normalize = T.MotCompose([
#         T.MotToTensor(),
#         T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

#     if image_set == 'train':
#         return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
#             T.MotRandomSelect(
#                 T.MotRandomResize(scales, max_size=1536),
#                 T.MotCompose([
#                     T.MotRandomResize([400, 500, 600]),
#                     T.FixedMotRandomCrop(384, 600),
#                     T.MotRandomResize(scales, max_size=1536),
#                 ])
#             ),
#             normalize,
#         ])

#     if image_set == 'val':
#         return T.MotCompose([
#             T.MotRandomResize([800], max_size=1333),
#             normalize,
#         ])

#     raise ValueError(f'unknown {image_set}')
###############################################################################

def make_transforms_for_applemots(image_set, args=None):
    
    ##########################################################################
    # () Defining custom cropping
    class CustomFixedRandomCrop:
        def __init__(self, crop_size, max_attempts=100000):
            self.crop_size = crop_size
            self.max_attempts = max_attempts
            
        def calculate_overlap(self, bbox, crop):
                # bbox is (x1, y1, x2, y2) and crop is (left, top, right, bottom)
                dx = min(bbox[2], crop[2]) - max(bbox[0], crop[0])
                dy = min(bbox[3], crop[3]) - max(bbox[1], crop[1])
                if (dx >= 0) and (dy >= 0):
                    overlap_area = dx * dy
                else:
                    overlap_area = 0

                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                return overlap_area / bbox_area if bbox_area > 0 else 0


        def __call__(self, images, all_targets):
            cropped_images = []
            cropped_targets_list = []
            image_idx = 0 
            # save_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main/output/customcropping"
            # title = f"{image_idx}.png"
            # save_image_with_bboxes_and_masks(images, all_targets, title, save_path)
            # output_dir = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main/output/center_points"
            

            for img_idx, (image, targets) in enumerate(zip(images, all_targets)):
                img_width, img_height = image.size
                crop_height, crop_width = self.crop_size
                crop_height = min(crop_height, img_height)
                crop_width = min(crop_width, img_width)
                # Convert the bounding boxes tensor to a list of tuples for indexing
                bbox_tuples = [tuple(box.tolist()) for box in targets['boxes']]
                valid_crop_found = False
                for attempt in range(self.max_attempts):
                    # Select a random bounding box to start the crop
                    bbox_idx = random.randint(0, len(targets['boxes']) - 1)
                    selected_bbox = targets['boxes'][bbox_idx]

                    # Calculate the maximum top and left coordinates for the crop
                    top_max = min(selected_bbox[1].int().item(), img_height - crop_height)
                    left_max = min(selected_bbox[0].int().item(), img_width - crop_width)

                    # Randomly choose the top and left coordinates for the crop
                    top = random.randint(0, top_max)
                    left = random.randint(0, left_max)

                    # Calculate the bottom and right coordinates for the crop
                    bottom = top + crop_height
                    right = left + crop_width
                    
                    overlap = self.calculate_overlap(selected_bbox, (left, top, right, bottom))
                    # Check if the selected bounding box is completely within the crop area
                    if overlap >= 0.4:
                        valid_crop_found = True
                        image_cropped = F.crop(image, top, left, crop_height, crop_width)

                        # Process bounding boxes and masks
                        cropped_bboxes = []
                        cropped_masks = []
                        cropped_labels = []
                        cropped_obj_ids = []
                        cropped_areas = []
                        cropped_iscrowd = []
                        for bbox, mask in zip(targets['boxes'], targets['masks']):
                            # Adjust bounding boxes and masks to fit the crop
                            new_bbox = [max(bbox[0] - left, 0), max(bbox[1] - top, 0),
                                        min(bbox[2] - left, crop_width), min(bbox[3] - top, crop_height)]
                            # Check if the bbox has a valid size
                            if new_bbox[2] > new_bbox[0] and new_bbox[3] > new_bbox[1]:
                                # Convert bbox tensor to tuple for indexing
                                bbox_tuple = tuple(bbox.tolist())

                                # Find the index of bbox in the original targets['boxes']
                                if bbox_tuple in bbox_tuples:
                                    index = bbox_tuples.index(bbox_tuple)
                                else:
                                    continue  # Skip if bbox not found in original list
                                cropped_labels.append(targets['labels'][index])
                                cropped_obj_ids.append(targets['obj_ids'][index])
                                cropped_areas.append(targets['area'][index])
                                cropped_iscrowd.append(targets['iscrowd'][index])
                                cropped_bboxes.append(new_bbox)
                                # Crop the mask accordingly
                                mask_cropped = mask[top:bottom, left:right]
                                cropped_masks.append(mask_cropped)
                                
                        # Visualization code starts here
                        # fig, ax = plt.subplots()
                        # # ax.imshow(image_cropped.permute(1, 2, 0).numpy())  # Convert to numpy array for matplotlib
                        # ax.imshow(np.array(image_cropped))   # Convert to numpy array for matplotlib
                        # for bbox in cropped_bboxes:
                        #     # Create a Rectangle patch
                        #     rect = patches.Rectangle(
                        #         (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                        #         linewidth=1, edgecolor='r', facecolor='none')
                        #     ax.add_patch(rect)

                        # plt.axis('off')  # Hide the axis
                        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        # filename = f"cropped_image_{img_idx}_{timestamp}.png"
                        # full_save_path = os.path.join(output_dir, filename)
                        # plt.savefig(full_save_path, bbox_inches='tight', pad_inches=0)
                        # plt.close(fig)
                        
    
                        # Only save the crop if there's at least one valid bbox
                        if cropped_bboxes:
                            cropped_targets = {
                            'boxes': torch.tensor(cropped_bboxes, dtype=torch.float32),
                            'labels': torch.tensor(cropped_labels, dtype=torch.int64),
                            'obj_ids': torch.tensor(cropped_obj_ids, dtype=torch.int64),
                            'area': torch.tensor(cropped_areas, dtype=torch.float32),
                            'iscrowd': torch.tensor(cropped_iscrowd, dtype=torch.int64),
                            'masks': torch.stack(cropped_masks)
                            }
                            cropped_images.append(image_cropped)
                            cropped_targets_list.append(cropped_targets)
                            break  # Exit the attempt loop

                if not valid_crop_found:
                    top = (img_height - crop_height) // 2
                    left = (img_width - crop_width) // 2
                    image_cropped = F.crop(image, top, left, crop_height, crop_width)

                    # Get the current timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

                    # Save the image that did not contain any valid bbox after all attempts
                    save_failed_image_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask_AppleMOTS-InstanceAware/output/customcropping"  
                    failed_image_filename = f"failed_image_{img_idx}_{timestamp}.png"  # Include timestamp in the filename
                    failed_image_full_path = os.path.join(save_failed_image_path, failed_image_filename)

                    # Save the image
                    image_cropped.save(failed_image_full_path)

                    raise ValueError(f"Failed to find a valid crop after {self.max_attempts} attempts.")
                image_idx +=1
            return cropped_images, cropped_targets_list
    ##########################################################################
    
    save_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask_AppleMOTS-InstanceAware/output/data_aug"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    def save_image_with_bboxes_and_masks(images, targets, title, save_path):
        
        for img_idx, (img, target) in enumerate(zip(images, targets)):
            # Ensure img is a numpy array in the correct format (H, W, C) for RGB images
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            elif isinstance(img, Image.Image):
                img = np.array(img)
            if img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img_with_annotations = img.copy()
            # print('target in plot is:', target)
            for bbox, mask in zip(target['boxes'], target['masks']):
                # Convert the bounding box to integer pixel indices
                x_min, y_min, x_max, y_max = bbox.int().tolist()
                
                # Draw the bounding box as a rectangle
                cv2.rectangle(img_with_annotations, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Create a mask overlay with a colormap
                mask_overlay = np.ma.masked_where(mask == 0, mask).astype(np.uint8)
                colored_mask = plt.get_cmap('spring')(mask_overlay)[:, :, :3]  # Get only the RGB channels
                colored_mask = (colored_mask * 255).astype(np.uint8)  # Scale to 0-255 range
                # if img_with_annotations.shape[:2] != colored_mask.shape[:2]:
                #     colored_mask = resize_mask(colored_mask, img_with_annotations.shape[:2])

                # print(img_with_annotations.shape, colored_mask.shape)
                # print(img_with_annotations.shape)

                # Blend the colored mask with the image
                img_with_annotations = cv2.addWeighted(img_with_annotations, 1.0, colored_mask, 0.5, 0)
            
            
            # Generate a unique filename using the current timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            unique_filename = f"{title}_{img_idx}_{timestamp}.png"
            image_save_path = os.path.join(save_path, unique_filename)
            
            
            # Save the image with annotations
            # image_save_path = os.path.join(save_path, f"{title}_{img_idx}.png")
            cv2.imwrite(image_save_path, cv2.cvtColor(img_with_annotations, cv2.COLOR_RGB2BGR))

    
    # def debug_transform(targets, stage, timing, image):
    #     print(f"Debug {timing} {stage}:")
    #     for t in targets:
    #         if 'masks' in t and 'boxes' in t:
    #             print(f"Masks={len(t['masks'])}, Boxes={len(t['boxes'])}")
    #             # if len(t['masks']) != len(t['boxes']):
    #             #     plot_image_with_bboxes_and_masks(image, t['boxes'], t['masks'], f"{timing} {stage} Mismatch")
    
    def apply_transform_and_debug(transform, image, targets, stage, save_path):
        # print(f"--- Applying {stage} ---")
        # debug_transform(targets, stage, "before", image)

        if isinstance(transform, T.MotCompose):
            for sub_transform in transform.transforms:
                # print(f"--- Applying sub-transform {sub_transform.__class__.__name__} in {stage} ---")
                image, targets = sub_transform(image, targets)
                # debug_transform(targets, sub_transform.__class__.__name__, "after", image)
                
                # save_image_with_bboxes_and_masks(image, targets, f"{stage}_final", save_path)
               
        else:
            image, targets = transform(image, targets)
     
        # debug_transform(targets, stage, "after", image)
        
        # save_image_with_bboxes_and_masks(image, targets, f"{stage}_final", save_path)

        return image, targets
    
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        transforms_list = [
            ("MotRandomHorizontalFlip", T.MotRandomHorizontalFlip()),
            ("MotRandomResize", T.MotRandomResize(scales, max_size=1536)),
            ("MotCompose", T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    ####################################################
                    # () Adding a custom crop function
                    # T.FixedMotRandomCrop(384, 600),
                    CustomFixedRandomCrop((384, 600)),
                    ####################################################
                    T.MotRandomResize(scales, max_size=1536),
                ]))
        ]
        
        transforms_with_debug = []
        for stage, transform in transforms_list:
            wrapped_transform = lambda image, targets, transform=transform, stage=stage: apply_transform_and_debug(transform, image, targets, stage, save_path)
            transforms_with_debug.append(wrapped_transform)

        return T.Compose(transforms_with_debug + [normalize])
        

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
        

def make_transforms_for_crowdhuman(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.FixedMotRandomShift(bs=1),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,

        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    crowdhuman_train = make_transforms_for_crowdhuman('train', args)
    
    ###############################################################################
    # () Adding AppleMOTS
    applemots_train = make_transforms_for_applemots('train', args)
    applemots_val = make_transforms_for_applemots('val', args)
    ###############################################################################
    
    
    ###########################################################################################
    # () Adding AppleMOTS
    # dataset2transform_train = {'MOT17': mot17_train, 'CrowdHuman': crowdhuman_train}
    # dataset2transform_val = {'MOT17': mot17_test, 'CrowdHuman': mot17_test}
    
    dataset2transform_train = {'MOT17': mot17_train, 'CrowdHuman': crowdhuman_train, 'APPLE_MOTS':applemots_train}
    dataset2transform_val = {'MOT17': mot17_test, 'CrowdHuman': mot17_test, 'APPLE_MOTS':applemots_val}
     ###########################################################################################
    
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    return dataset


