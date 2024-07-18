# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
from util.evaluation import Evaluator
import motmetrics as mm
import shutil
from pycocotools import mask as mask_util
from models.structures import Instances
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as mask_utils
import pycocotools.mask as mask_utils
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from skimage.filters import threshold_otsu
from skimage.morphology import square
import pandas as pd
import datetime
from matplotlib import patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from util.thresholding import sauvola_threshold, niblack_threshold, nick_threshold, apply_threshold, apply_gaussian_filter
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from scipy.ndimage import label, find_objects

np.random.seed(2020)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class MOTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        pass

    def update(self, dt_instances: Instances):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label == 0:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.masks[i].flatten(), dt_instances.scores[i:i+1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))



def decode_RLE_to_mask(rle_str, h, w):
    rle = {
        'counts': rle_str,
        'size': [h, w]
    }
    mask = mask_utils.decode(rle)
    return mask

#


def load_label(combined_path: str, img_size: tuple) -> dict:
    targets = {'boxes': [], 'masks': [], 'area': [], 'labels': [], 'obj_ids' : []}
    h, w = img_size  # Image dimensions

    if osp.isfile(combined_path):
        # Load combined data (bbox + RLE mask)
        with open(combined_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id = parts[0] 
                obj_id= parts[1] 
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
                targets['labels'].append(0)  # Assuming single class for simplicity
                targets['obj_ids'].append(obj_id)

        # Convert lists to tensors
        targets['boxes'] = np.asarray(targets['boxes'], dtype=np.float32).reshape(-1, 4)
        targets['area'] = np.asarray(targets['area'], dtype=np.float32)
        targets['labels'] = np.asarray(targets['labels'], dtype=np.int64)
        targets['masks'] = np.stack([torch.from_numpy(np.array(m)) for m in targets['masks']])
        targets['obj_ids'] = np.asarray(targets['obj_ids'], dtype=np.int64)
    else:
        raise ValueError('Invalid path provided: ' + combined_path)
    return targets


def filter_pub_det(res_file, pub_det_file, filter_iou=False):
    frame_boxes = {}
    with open(pub_det_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(',')
            frame_id = int(elements[0])
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
            frame_boxes[frame_id].append([x1, y1, x2, y2])

    for frame, boxes in frame_boxes.items():
        frame_boxes[frame] = np.array(boxes)

    ids = {}
    num_filter_box = 0
    with open(res_file, 'r') as f:
        lines = list(f.readlines())
    with open(res_file, 'w') as f:
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(',')
            frame_id, obj_id = elements[:2]
            frame_id = int(frame_id)
            obj_id = int(obj_id)
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if obj_id not in ids:
                # track initialization.
                if frame_id not in frame_boxes:
                    num_filter_box += 1
                    print("filter init box {} {}".format(frame_id, obj_id))
                    continue
                pub_dt_boxes = frame_boxes[frame_id]
                dt_box = np.array([[x1, y1, x2, y2]])
                if filter_iou:
                    max_iou = bbox_iou(dt_box, pub_dt_boxes).max()
                    if max_iou < 0.5:
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                else:
                    pub_dt_centers = (pub_dt_boxes[:, :2] + pub_dt_boxes[:, 2:4]) * 0.5
                    x_inside = (dt_box[0, 0] <= pub_dt_centers[:, 0]) & (dt_box[0, 2] >= pub_dt_centers[:, 0])
                    y_inside = (dt_box[0, 1] <= pub_dt_centers[:, 1]) & (dt_box[0, 3] >= pub_dt_centers[:, 1])
                    center_inside: np.ndarray = x_inside & y_inside
                    if not center_inside.any():
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                print("save init track {} {}".format(frame_id, obj_id))
                ids[obj_id] = True
            f.write(line)

    print("totally {} boxes are filtered.".format(num_filter_box))


class Detector(object):
    def __init__(self, args, model=None, seq_num=2):

        self.args = args
        self.detr = model

        self.seq_num = seq_num
        img_list = os.listdir(os.path.join(self.args.mot_path, 'MOTS/train/images', self.seq_num, 'img1'))
        img_list = [os.path.join(self.args.mot_path, 'MOTS/train/images', self.seq_num, 'img1', _) for _ in img_list if
                    ('jpg' in _) or ('png' in _)]
        # img_list = os.listdir(os.path.join(self.args.mot_path, 'MOTS/test/images', self.seq_num, 'img1'))
        # img_list = [os.path.join(self.args.mot_path, 'MOTS/test/images', self.seq_num, 'img1', _) for _ in img_list if
        #             ('jpg' in _) or ('png' in _)]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)
        self.tr_tracker = MOTR()

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.mask_statistics = pd.DataFrame(columns=['frame_id', 'track_id', 'mask_max', 'mask_min', 'mask_mean'])
        
        self.save_path = os.path.join(self.args.output_dir, 'results/{}'.format(seq_num))
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(self.args.output_dir, 'preds', self.seq_num)
        os.makedirs(self.predict_path, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_path,  'gt.txt')):
            os.remove(os.path.join(self.predict_path,  'gt.txt'))

    
    def load_img_from_file(self, f_path): 
        
        # def visualize_annotations(image, targets):
        #     fig, ax = plt.subplots(1, figsize=(12, 9))
        #     ax.imshow(image)
        #     # If targets is None or empty, just show the image
        #     if not targets:
        #         plt.show()
        #         return
        #     # Otherwise, loop through the targets and plot them
        #     for box, mask in zip(targets['boxes'], targets['masks']):
        #         # Draw the bounding box
        #         # rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='b', facecolor='none')
        #         # ax.add_patch(rect)
        #         # Overlay the mask on the image
        #         color_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        #         color_mask[mask > 0] = [0 , 255 , 0, 127]  # Green with half transparency
        #         # Overlay the color mask on the image
        #         ax.imshow(color_mask)
        #     plt.axis('off')
        #     plt.show()
        

        # def visualize_annotations(image, targets):
        #     fig, ax = plt.subplots(1, figsize=(12, 9))
        #     ax.imshow(image)

        #     if not targets:
        #         plt.show()
        #         return

        #     unique_ids = np.unique(targets['obj_ids'])  # Get unique IDs to ensure colormap consistency
        #     cmap = plt.cm.get_cmap('tab20', len(unique_ids))
        #     norm = plt.Normalize(vmin=0, vmax=len(unique_ids))

        #     for box, mask, obj_id in zip(targets['boxes'], targets['masks'], targets['obj_ids']):
        #         index = np.where(unique_ids == obj_id)[0][0]  # Find the index of obj_id in unique_ids
        #         color = cmap(norm(index))
        #         color_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        #         color_mask[mask > 0] = np.array(color) * 255
        #         ax.imshow(color_mask)

        #     plt.axis('off')
        #     plt.show()

        def visualize_annotations(image, targets, output_dir):
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            fig, ax = plt.subplots(1, figsize=(12, 9))
            ax.imshow(image)

            if not targets:
                # Save the plain image if no targets are present
                filepath = os.path.join(output_dir, datetime.now().strftime("plain_image_%Y%m%d_%H%M%S.png"))
                plt.savefig(filepath)
                plt.close(fig)
                return

            unique_ids = np.unique(targets['obj_ids'])  # Get unique IDs to ensure colormap consistency
            cmap = cm.get_cmap('tab20', len(unique_ids))
            norm = Normalize(vmin=0, vmax=len(unique_ids))

            for box, mask, obj_id in zip(targets['boxes'], targets['masks'], targets['obj_ids']):
                index = np.where(unique_ids == obj_id)[0][0]  # Find the index of obj_id in unique_ids
                color = cmap(norm(index))
                color_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                color_mask[mask > 0] = np.array(color) * 255
                ax.imshow(color_mask)

            plt.axis('off')

            # Save the annotated image
            filepath = os.path.join(output_dir, datetime.datetime.now().strftime("annotated_image_%Y%m%d_%H%M%S.png"))
            plt.savefig(filepath)
            plt.close(fig)
        
        #For test
        label_path = f_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        cur_img = cv2.imread(f_path)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        img_size = (cur_img.shape[0], cur_img.shape[1])
        
        #For test
        # targets = load_label(label_path, img_size) 
        # visualize_annotations(cur_img, targets, '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/output/MOTS_gt/seq11')
        # return cur_img, targets
        return cur_img

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
        
    @staticmethod
    def write_results(txt_path, mask_statistics, frame_id, bbox_xyxy, masks, identities):
        processed_masks = []  
        threshold_iou = 0.1   
        
        def iou_mask(mask1, mask2):
            """Calculate the Intersection over Union (IoU) of two binary masks."""
            intersection = np.logical_and(mask1, mask2)
            union = (np.logical_or(mask1, mask2))
            iou = (np.sum(intersection) / np.sum(union))
            return iou
        
        def safe_iou(pred_mask, gt_mask):
            # Calculate intersection and union
            intersection = np.logical_and(pred_mask, gt_mask)
            union = np.logical_or(pred_mask, gt_mask)
            
            # Sum the areas
            intersection_sum = np.sum(intersection)
            union_sum = np.sum(union)
            
            # Check for zero union case
            if union_sum == 0:
                return 0.0  # Return an IoU of 0 if there's no union; alternative approaches could be used based on context
            else:
                return intersection_sum / union_sum
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def plot_mask(mask, title):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            plt.imshow(mask, cmap='gray')
            plt.title(title)
            plt.axis('off')
            plt.show()
    
    
        def encode_mask_to_RLE_results(binary_mask):
            if isinstance(binary_mask, torch.Tensor):
                # Convert to a NumPy array and ensure it's uint8
                binary_mask = binary_mask.cpu().numpy().astype(np.uint8)
            else:
                # If it's already a NumPy array, just ensure it's the correct type
                binary_mask = binary_mask.astype(np.uint8)
            fortran_binary_mask = np.asfortranarray(binary_mask)
            rle = mask_utils.encode(fortran_binary_mask)
            return rle['counts'].decode('ascii')
        
        
        # Prepare to collect statistics
        # mask_statistics = pd.DataFrame(columns=['frame_id', 'track_id', 'mask_max', 'mask_min', 'mask_mean'])
        # save_format = '{frame},{id},{mask_height},{mask_width},{mask_rle},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        save_format = '{frame},{id},{class_id},{mask_height},{mask_width},{mask_rle}\n'
        with open(txt_path, 'a') as f:
            for xyxy, mask, track_id in zip(bbox_xyxy, masks, identities):
                
                # Collect data for histogram
                # mask_max = mask.max()
                # mask_min = mask.min()
                # mask_mean = mask.mean() 
                
                # Append the stats to the DataFrame
                # mask_statistics = mask_statistics.append({
                #     'frame_id': frame_id,
                #     'track_id': track_id,
                #     'mask_max': mask_max,
                #     'mask_min': mask_min,
                #     'mask_mean': mask_mean
                # }, ignore_index=True)
        
                
                if track_id < 0 or track_id is None:
                    continue

                # Adaptive threshold
                smooth_mask = gaussian_filter(mask, sigma=1)
                # smooth_mask = apply_gaussian_filter(mask, sigma=1)
                # Exclude zero values for thresholding calculation
                # nonzero_mask_values = smooth_mask[smooth_mask > 0]
                # if nonzero_mask_values.size > 0:
                #     optimal_threshold = threshold_otsu(nonzero_mask_values)
                # else:
                #     # Default threshold if mask has no non-zero values
                #     optimal_threshold = 0.5
                
                # # Apply threshold
                # binary_mask = smooth_mask > optimal_threshold
    
                    
                # Hard threshold
                # mask = smooth_mask>0.3

                # Train
                # smooth_mask = gaussian_filter(mask, sigma=1)
                # mask = smooth_mask>0.6
                # mask = apply_adaptive_threshold(mask)

                # # Train: Adaptive local threshold
                # thresh_sauvola = threshold_sauvola(smooth_mask, window_size=25)
                # valid_thresholds = thresh_sauvola[thresh_sauvola > 0.001] 

                # if valid_thresholds.size > 0:
                #     # If there are valid values, replace thresholds <= 0.001 with the minimum valid value
                #     min_valid_value = np.min(valid_thresholds)  # ensure only valid values are considered
                #     thresh_sauvola[thresh_sauvola <= 0.001] = min_valid_value
                # else:
                #     thresh_sauvola[:] = 0.6

                # binary_mask = smooth_mask > thresh_sauvola
                
                # # Label connected components
                # labeled_array, num_features = label(binary_mask)

                # # Measure sizes of components
                # sizes = np.bincount(labeled_array.ravel())
                # mask_sizes = sizes > 1150  # 1080*1920
                # # mask_sizes = sizes > 200  # 640*480
                # mask_sizes[0] = 0  # Background size (zero) must not be removed

                # # Apply the mask to the labeled array
                # connected_component_mask = mask_sizes[labeled_array]
                
                # # Morphological opening and closing
                # struct = generate_binary_structure(2, 1)
                # opened_mask = binary_opening(connected_component_mask, structure=struct)
                # mask = binary_closing(opened_mask, structure=struct)
                
                # Test: Adaptive local threshold
                thresh_sauvola = threshold_sauvola(smooth_mask, window_size=25)
                valid_thresholds = thresh_sauvola[thresh_sauvola > 0.1] 

                if valid_thresholds.size > 0:
                    # If there are valid values, replace thresholds <= 0.001 with the minimum valid value
                    min_valid_value = np.min(valid_thresholds)  # ensure only valid values are considered
                    thresh_sauvola[thresh_sauvola <= 0.1] = min_valid_value
                else:
                    thresh_sauvola[:] = 0.3

                # binary_mask = smooth_mask > thresh_sauvola
                binary_mask = smooth_mask > 0.3
                
                # Manual threshold onnected components
                # labeled_array, num_features = label(binary_mask)
                # sizes = np.bincount(labeled_array.ravel())
                # mask_sizes = sizes > 1800  # 1080*1920
                # # mask_sizes = sizes > 200  # 640*480
                # mask_sizes[0] = 0  # Background size (zero) must not be removed
                # connected_component_mask = mask_sizes[labeled_array]
                
                # Adaptive threshold onnected components
                labeled_array, num_features = label(binary_mask)
                component_slices = find_objects(labeled_array)
                # component_areas = [labeled_array[s].size for s in component_slices]
                component_areas = np.bincount(labeled_array.ravel())[1:]
                if component_areas.size > 0:
                    largest_component_index = np.argmax(component_areas) + 1  # +1 because labels start from 1
                    connected_component_mask = (labeled_array == largest_component_index)
                else:
                    connected_component_mask = binary_mask
                    
                # Morphological opening and closing
                struct = generate_binary_structure(2, 2)
                opened_mask = binary_opening(connected_component_mask, structure=struct)
                mask = binary_closing(opened_mask, structure=struct)

                
                # plt.figure(figsize=(8, 7))
                # plt.subplot(2, 2, 1)
                # plt.imshow(binary_mask)
                # plt.title('sauvola')
                # plt.axis('off')
                
                
                # plt.subplot(2, 2, 2)
                # plt.imshow(connected_component_mask)
                # plt.title('connected_component_mask')
                # plt.axis('off')
                
                
                # plt.subplot(2, 2, 3)
                # plt.imshow(opened_mask)
                # plt.title('opened_mask')
                # plt.axis('off')
                
                # plt.subplot(2, 2, 4)
                # plt.imshow(mask)
                # plt.title('Connected Component')
                # plt.axis('off')
                
                # plt.show()
                
                # Check for duplicates
                # is_duplicate = any(iou_mask(mask, pm) > threshold_iou for pm in processed_masks)
                is_duplicate = any(safe_iou(mask, pm) > threshold_iou for pm in processed_masks)
                if not is_duplicate:
                    processed_masks.append(mask)
                    class_id = 2
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    mask_height, mask_width= mask.shape[0], mask.shape[1]
                    mask_rle = encode_mask_to_RLE_results(mask)
                    # plot_mask(mask, f"Track ID: {track_id}, Frame ID: {frame_id}")
                    
                    line = save_format.format(frame=int(frame_id), id=int(track_id), class_id = int(class_id), mask_height=int(mask_height),mask_width=int(mask_width),mask_rle=mask_rle)
                    f.write(line)
                # else:
                #     print('iou_mask(mask, pm)')
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Append the timestamp to your filename
        # filename = f'/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/distribution/mask_statistics_{current_time}.csv'
        # mask_statistics.to_csv(filename, index=False)

    def eval_seq(self):
        data_root = os.path.join(self.args.mot_path, 'MOTS/train/images')
        # data_root = os.path.join(self.args.mot_path, 'MOTS/test/images')
        # print("Self.predict_path is:", self.predict_path)
        result_filename = os.path.join(self.predict_path, 'gt.txt')
        evaluator = Evaluator(data_root, self.seq_num)
        # print('result_filename is:', result_filename)
        accs = evaluator.eval_file(result_filename)
        return accs

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        cv2.imwrite(img_path, img_show)

    # def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):
    def detect(self, prob_threshold=0.8, area_threshold=100, vis=False):
        total_dts = 0
        track_instances = None
        max_id = 0
        for i in tqdm(range(0, self.img_len)):
            # img, targets = self.load_img_from_file(self.img_list[i])
            img= self.load_img_from_file(self.img_list[i])
            cur_img, ori_img = self.init_img(img)
            # print('img:', ori_img.shape)

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
                track_instances.remove('masks')
                # Integrating masks
                # track_instances.remove('masks')

            res = self.detr.inference_single_image(cur_img.cuda().float(), (self.seq_h, self.seq_w), track_instances)
            track_instances = res['track_instances']
            # print('track_instances[masks]:', track_instances.masks.shape)
            max_id = max(max_id, track_instances.obj_idxes.max().item())

            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            dt_instances = track_instances.to(torch.device('cpu'))

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
            
            total_dts += len(dt_instances)

            # if vis:
            #     # for visual
            #     cur_vis_img_path = os.path.join(self.save_path, 'frame_{}.jpg'.format(i))
            #     gt_boxes = None
            #     self.visualize_img_with_bbox(cur_vis_img_path, ori_img, dt_instances, ref_pts=all_ref_pts, gt_boxes=gt_boxes)

            tracker_outputs = self.tr_tracker.update(dt_instances)
            # print("tracker_outputs[:, 4:-2]:", tracker_outputs[:, 4:-2].shape, "tracker_outputs[:, :4]:", tracker_outputs[:, :4], "tracker_outputs[:, -1]:", tracker_outputs[:, -1])
            # For MOTS20-05 & MOTS20-06
            img_h , img_w = ori_img.shape[0], ori_img.shape[1]
            self.write_results(txt_path=os.path.join(self.predict_path, 'gt.txt'),
                               mask_statistics = self.mask_statistics,
                               frame_id=(i + 1),
                               bbox_xyxy=tracker_outputs[:, :4],
                               masks = tracker_outputs[:, 4:-2].reshape(-1, img_h , img_w) ,
                               identities=tracker_outputs[:, -1])
            
            # For MOTS20-02/09/11
            # self.write_results(txt_path=os.path.join(self.predict_path, 'gt.txt'),
            #                    mask_statistics = self.mask_statistics,
            #                    frame_id=(i + 1),
            #                    bbox_xyxy=tracker_outputs[:, :4],
            #                    masks = tracker_outputs[:, 4:-2].reshape(-1, 1080, 1920) ,
            #                    identities=tracker_outputs[:, -1])
        print("totally {} dts max_id={}".format(total_dts, max_id))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr = detr.cuda()
    detr.eval()
    
    # seq_nums = ['MOTS20-05']
    # seq_nums = ['MOTS20-06']
    
    seq_nums = ['MOTS20-02',
                'MOTS20-05',
                'MOTS20-09',
                'MOTS20-11',]
    # seq_nums = ['MOTS20-01',
    #             'MOTS20-06',
    #             'MOTS20-07',
    #             'MOTS20-12',]
   


    accs = []
    seqs = []

    for seq_num in seq_nums:
        print("solve {}".format(seq_num))
        det = Detector(args, model=detr, seq_num=seq_num)
        det.detect(vis=True)
        # accs.append(det.eval_seq())
        # print('det.eval_seq():', det.eval_seq())
        seqs.append(seq_num)

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    with open("eval_log.txt", 'a') as f:
        print(strsummary, file=f)

