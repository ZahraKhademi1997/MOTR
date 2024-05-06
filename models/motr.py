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
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
from util.box_ops import masks_to_boxes
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
####################################################################################
# Importing libraries for instance segmentation head
from models.mmcv_utils.mask_position_encoding import build_positional_encoding
from models.mmcv_utils.dynamicdeformableattention import DynamicDeformableAttention
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
####################################################################################
from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, MHAttentionMap, MaskHeadSmallConv, PerPixelEmbedding,
                           aligned_bilinear, dice_loss, sigmoid_focal_loss, focal_loss, generalized_dice_loss, dual_focal_loss)
from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
import matplotlib.pyplot as plt
import code
from datetime import datetime
import os
from pathlib import Path
from PIL import Image


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        
        # (5) Adding some notations for the forward function
        self.mask_height = None
        self.mask_width = None
        

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            
            # (23) Adding loss_masks to dictionary
            'masks': self.loss_masks,
            
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses
    
    
    # (2) Adding loss_mask from deformable_detr to motr     
    def loss_masks(self, outputs, gt_instances: List[Instances], indices, num_boxes):
        
        def save_image(feature_map, layer_name):
            image_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-AppleMots-MaskFormer-Instance/output/pred_masks/loss_masks_py"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i in range(feature_map.size(0)):
                plt.imshow(feature_map[i, 0].detach().cpu().numpy(), cmap='gray')
                plt.title(f"{layer_name}_{i}")
                filename = f"{layer_name}_{i}_{timestamp}.png"
                plt.savefig(os.path.join(image_path, filename))
                plt.close()
                
        def save_mask_with_boxes(gt_instances, target_masks, indices, layer_name):
            image_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-AppleMots-MaskFormer-Instance/output/pred_masks/best_mask_boxes"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            _, box_ids = idx
            height, width = target_masks[0].shape[-2:]  # Get the spatial dimensions from the first target mask
            # Initialize a composite mask with the same height and width, and ensure it's on the same device and dtype
            composite_mask = torch.zeros((height, width), dtype=torch.bool, device=target_masks[0].device)
            for t in gt_instances:
                masks_field = t.get('masks')  # Assuming this is already a tensor
                # print('mask_fields:', masks_field.shape)
                for object_mask in masks_field:
                    object_mask_bool = object_mask.bool()  # Convert to boolean for logical operations
                    composite_mask = composite_mask | object_mask_bool
                    # print('composited_mask shape:', composite_mask.shape)
            feature_map = composite_mask.float().unsqueeze(0)
        
            # Convert the tensor to a numpy array and squeeze in case there is an extra dimension
            mask_np = feature_map.squeeze(0).detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(12, 8))  # You can adjust the figure size as needed
            ax.imshow(mask_np, cmap='gray')

            img_height, img_width = mask_np.shape

            for i, box in enumerate(src_boxes):
                # Convert box coordinates from normalized to image pixel coordinates
                # x1, y1, x2, y2 = box
                # x1 = int(x1 * width)
                # y1 = int(y1 * height)
                # x2 = int(x2 * width)
                # y2 = int(y2 * height)
                
                x_center, y_center, width, height = box
                x_min = max(int((x_center - width / 2) * img_width), 0)
                y_min = max(int((y_center - height / 2) * img_height), 0)
                x_max = min(int((x_center + width / 2) * img_width), img_width - 1)
                y_max = min(int((y_center + height / 2) * img_height), img_height - 1)
                
                # Extract the region of the best_pred_mask that the current src_box covers
                region = mask_np[y_min:y_max+1, x_min:x_max+1]
                
                # Get the corresponding box ID
                box_id = box_ids[i].item()

                # Draw the rectangle on the image
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # Annotate the box with its ID
                ax.text(x_min, y_min, str(box_id), verticalalignment='top', color='white', fontsize=8, weight='bold')
            
            plt.title(f"{layer_name}")
            filename = f"{layer_name}_{timestamp}.png"
            plt.savefig(os.path.join(image_path, filename))
            plt.close()

                
        """Compute the losses related to the masks: the focal loss and the dice loss.
            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks = outputs["pred_masks"].requires_grad_(True)
        src_masks = outputs["pred_masks"]
        # print('src_masks before:', src_masks.shape) #src_masks before: torch.Size([1, 300, 960, 1500])
        src_masks = src_masks[src_idx]
        # print('src_masks after:', src_masks.shape) # src_masks after: torch.Size([23, 960, 1500])
        
        masks = []
        # print(gt_instances)
        for t in gt_instances:
            masks_field = t.get('masks')
            masks.append(masks_field)
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        size = target_masks.shape[-2:]
        # save_mask_with_boxes(gt_instances, target_masks,indices,'gt_mask')
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="nearest")
        # save_image(src_masks, 'loss_masks')
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            # "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, size, num_boxes),
            # "loss_mask": dual_focal_loss(src_masks, target_masks, num_boxes),
            # "loss_dice": generalized_dice_loss(src_masks, target_masks, num_boxes),
            # "loss_mask": focal_loss(src_masks, target_masks, num_boxes),
        }
        
        return losses
    
    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
        
        # (11) Adding pred_masks
        pred_masks_i = track_instances.pred_masks  # predicted masks of i-th image.
        
        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
            
            # (12) Adding pred_masks
            'pred_masks': pred_masks_i.unsqueeze(0),
            
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            
            # (13) Solving device problem in qim
            # new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
            #                                   dim=1).to(pred_logits_i.device)
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
            
            # (14) Adding pred_masks
            'pred_masks': track_instances.pred_masks[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        
        # (15) Adding active track masks
        active_track_masks = track_instances.pred_masks[active_idxes]
        active_track_masks = active_track_masks.sigmoid()
        
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            
            # (17) Adding active track masks
            # track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
            track_instances.iou_boxes[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
           
        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes
        
        # (18) Adding out_masks
        out_mask = track_instances.pred_masks

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]
        
        # (19) Adding out_masks
        masks = out_mask.squeeze(0)
        track_instances.masks = masks

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        
        # (20) Adding out_masks
        track_instances.remove('pred_masks')
        
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed, num_seg_fcs, dynamic_params_dims, dynamic_encoder_heads, mask_positional_encoding_cfg={'type': 'RelSinePositionalEncoding', 'num_feats': 4, 'normalize': True},
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        mask_hidden_dim = 8
        mask_nheads = 4
        # Adding SOIT segmentation head attributes
        self.num_seg_fcs = num_seg_fcs
        self.dynamic_params_dims = dynamic_params_dims
        self.dynamic_encoder_heads = dynamic_encoder_heads
        self.mask_positional_encoding_cfg = mask_positional_encoding_cfg
        
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
                
        ##############################################
        # (1) Adding segmentation head
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
    
        seg_branch = []
        for _ in range(self.num_seg_fcs):
            seg_branch.append(Linear(hidden_dim, hidden_dim))
            seg_branch.append(nn.ReLU())
        seg_branch.append(Linear(hidden_dim, self.dynamic_params_dims))
        seg_branch = nn.Sequential(*seg_branch)
        self.seg_branches = nn.ModuleList(
            [seg_branch for _ in range(num_pred)])
        
        self.mask_positional_encoding = build_positional_encoding(
            self.mask_positional_encoding_cfg)
        
        self.dynamic_encoder = DynamicDeformableAttention(
            embed_dims=mask_hidden_dim,
            num_heads=mask_nheads)
        ##############################################
        
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])
        track_instances.query_pos = self.query_embed.weight
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        
        # (16) Adding part to handle iou from masks and boxes
        # track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.iou_boxes = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.iou_masks = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        
        # (4) Initializing pred_masks in track_instances dictionary
        height = self.mask_height
        width = self.mask_width
        track_instances.pred_masks = torch.zeros((len(track_instances), height, width), dtype=torch.float, device=device)
        
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        bs = features[-1].tensors.shape[0]
        assert mask is not None
        
        # Loading the pre-defined queries from query_embedding.
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        
        img_size = samples.tensors.shape[-2:]
        img_masks = samples.mask.new_ones((bs, *img_size))
        # print('samples.tensors.shape:',samples.tensors.shape)
        for b in range(bs):
            img_h, img_w = samples.tensors.shape[2], samples.tensors.shape[3]
            img_masks[b, :img_h, :img_w] = 0
        
        
        srcs = []
        masks = []
        # mlvl_positional_encodings = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # print('mask shape backbone:', mask.shape)
            # print('src shape backbone:', src.shape)
            # pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            # pos.append(pos_l)
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # print('src shape:', src.shape)
                # print('mask shape:', mask.shape)
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                
        img_masks_float = img_masks.float()
        # Interpolate the float mask to the desired size
        interpolated_mask_float = F.interpolate(
            img_masks_float[None],  # Add batch dimension
            size=srcs[0].shape[-2:],  # Use the spatial dimensions of srcs[0]
            mode='nearest'  # Use nearest neighbor interpolation to avoid non-integer results
        )
        # Convert interpolated float mask back to boolean
        p3_mask = interpolated_mask_float.to(torch.bool).squeeze(0)  # Remove batch dimension and convert to bool
        # Assign the processed mask to self.p3_mask
        self.p3_mask = p3_mask
        # print('self.p3_mask:', self.p3_mask.shape) # torch.Size([1, 104, 145])

        
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory= self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, mask_proto= self.transformer(srcs, masks, pos , track_instances.query_posو query_embeds, ref_pts=track_instances.ref_pts)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, mask_proto = self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)

       
        (seg_memory, seg_pos_embed, seg_mask, spatial_shapes, seg_reference_points, level_start_index, valid_ratios) = mask_proto
        # print('seg_mask shape:', seg_mask.shape) # torch.Size([1, 22560])
        # print('seg_pos_embed:', seg_pos_embed.shape) #torch.Size([22560, 1, 256])
        # print('hs shape:', hs.shape) #torch.Size([6, 1, 200, 256])
        
        outputs_classes = []
        outputs_coords = []
        outputs_dynamic_params = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            # Adding dynamic parameter
            dynamic_params = self.seg_branches[lvl](hs[lvl])
            
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_dynamic_params.append(dynamic_params) # Query embedding
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_dynamic_params = torch.stack(outputs_dynamic_params)

        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        
        # print('ref_pts_all:', ref_pts_all.shape) # torch.Size([7, 1, 200, 2])
        # print('outputs_class:', outputs_class.shape) # torch.Size([6, 1, 200, 2])
        # print('outputs_coord:', outputs_coord.shape) # torch.Size([6, 1, 200, 4])
        # print('outputs_dynamic_params:', outputs_dynamic_params.shape) # torch.Size([6, 1, 200, 441])
        # print('hs:', hs.shape) # torch.Size([6, 1, 200, 256])
        # print('outputs_dynamic_params:', outputs_dynamic_params) # Contains negative values
        
        
        num_imgs = outputs_class[-1].size(0)
        # print('num_imgs:', num_imgs) #Batch size
        cls_scores_list = [outputs_class[-1][i] for i in range(num_imgs)]
        
        for i in range(num_imgs):
            mask_preds = []
            pos_dynamic_params = outputs_dynamic_params[-1][i]
            # print('pos_dynamic_params:', pos_dynamic_params.shape) #torch.Size([200, 441])
            pos_bbox_preds = outputs_coord[-1][i]
            pos_cxcy_coord = pos_bbox_preds[:, :2]
            img_mask_new = self.p3_mask[[i]]
            
            if pos_dynamic_params.size(0) > 0:
                for j in range(pos_dynamic_params.size(0)):
                    # print('pos_cxcy_coord[j]:', pos_cxcy_coord[j].shape) # torch.Size([2])
                    seg_pos_embed = self.mask_positional_encoding(
                        img_mask_new, pos_cxcy_coord[j])
                    # print('seg_pos_embed before:', seg_pos_embed.shape) # torch.Size([1, 8, 120, 188])
                    seg_pos_embed = seg_pos_embed.flatten(2).transpose(
                        1, 2).permute(1, 0, 2) 
                    
                    # print("seg_pos_embed min:", seg_pos_embed.min().item()) # -0.9999465346336365
                    # print("seg_pos_embed max:", seg_pos_embed.max().item()) # 1.0
                    # print("seg_pos_embed mean:", seg_pos_embed.mean().item()) # 0.2511863708496094

                    # print('seg_pos_embed after:', seg_pos_embed.shape) # torch.Size([22560, 1, 8])
                    # print('seg_memory[:, [i], :]:', seg_memory[:, [i], :].shape) #torch.Size([22560, 1, 8])
                    # print('seg_mask[[i]]:', seg_mask[[i]].shape) #torch.Size([1, 22560])
                    # print('seg_reference_points[[i]]:', seg_reference_points[[i]].shape) #torch.Size([1, 22560, 1, 2])
                    mask_preds.append(self.dynamic_encoder(
                        pos_dynamic_params[j],
                        seg_memory[:, [i], :],
                        None,
                        None,
                        query_pos=seg_pos_embed,
                        key_padding_mask=seg_mask[[i]],
                        reference_points=seg_reference_points[[i]],
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index))
                h, w = spatial_shapes[0]
                mask_preds = [
                    mask.squeeze().reshape(h, w) for mask in mask_preds]
                mask_preds = torch.stack(mask_preds)
                mask_preds = F.relu(mask_preds)
                # print('mask_preds before:', mask_preds.shape)
                # print('mask_preds:', mask_preds)
                # pad_mask = seg_mask[i].reshape(1, 1, h, w).float()
                # pad_mask = F.interpolate(
                #     pad_mask,
                #     scale_factor=4,
                #     mode='bilinear',
                #     align_corners=True).to(torch.bool).squeeze()
                # mask_preds = aligned_bilinear(
                #     mask_preds[None], factor=4)[0].sigmoid()
                # mask_preds.masked_fill(pad_mask, 0)
                # print('mask_preds after:', mask_preds.shape)
 
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5], 'pred_masks': mask_preds}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out
        
    def _post_process_single_image(self, frame_res, track_instances, is_last):
        ############################################################################################################################################## 
        def interpolation_masks(pred_masks):
            max_h, max_w = 972, 1296
            pred_masks = torch.nn.functional.interpolate(pred_masks.unsqueeze(0), size=(max_h, max_w), mode='nearest').squeeze(0)
            return pred_masks
        ##############################################################################################################################################   
                       
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()
        
        # pred_masks = frame_res['pred_masks'][0]
        # max_h, max_w = 972, 1296
        # pred_masks_interpolated = torch.nn.functional.interpolate(pred_masks.unsqueeze(0), size=(max_h, max_w), mode='nearest').squeeze(0)
             
        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        
        # (10) Adding pred_masks
        track_instances.pred_masks = interpolation_masks(frame_res['pred_masks'])
        # track_instances.pred_masks = frame_res['pred_masks'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        
        
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
            
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
            
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        # print('track_instances:', track_instances)
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
            # save_masks(frame_res['pred_masks'].squeeze(0), '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_test/output/motr_forward')
        else: 
            frame_res['track_instances'] = None
        
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        res = self._forward_single_image(img,
                                         track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        
        # Apply sigmoid to convert mask logits to probabilities
        if hasattr(track_instances, 'pred_masks'):
            mask_threshold = 0.5
            mask_probs = track_instances.pred_masks.sigmoid()
            
            # Apply threshold to convert probabilities to binary masks
            track_instances.pred_masks = (mask_probs > mask_threshold).float()
        
        
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        
        # (6) Calculating masks attribute for each image for _generate_empty_tracks function
        self.mask_height , self.mask_width = (972, 1296)
        
        
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            # (7) Adding pred_masks
            'pred_masks': [],   
        }

        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())
        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            if self.use_checkpoint and frame_index < len(frames) - 2:
                print('Entered if in forward of MOTR')
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        
                        # (8) Adding pred_masks to frame_res
                        frame_res['pred_masks'],
                        
                        frame_res['ref_pts'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'hs': tmp[3],
                    'aux_outputs': [{
                        'pred_logits': tmp[4+i],
                        'pred_boxes': tmp[4+5+i],
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)
            # print('frame_res:', frame_res)
            
            def save_masks(final_masks_tensor, output_dir):
                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Loop through each mask in the tensor
                for i, mask in enumerate(final_masks_tensor):
                    # Convert the mask to a NumPy array
                    mask_array = mask.detach().cpu().numpy()

                    # Generate a unique filename for each mask
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    filename = f'mask_{i}_{timestamp}.png'

                    # Save the mask as an image
                    plt.imsave(os.path.join(output_dir, filename), mask_array, cmap='gray')
                    
            track_instances = frame_res['track_instances']
            # Checking if the pred_masks are in corresct shape
            # save_masks(frame_res['pred_masks'].squeeze(0), '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-mask-AppleMots-ConnectedComponents/output/mask_forward_function')
            
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            # (9) Adding mask
                
            outputs['pred_masks'].append(frame_res['pred_masks'])
            

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 2, # Changing from 1 to 2 to include the background for segmentation
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    
    num_seg_fcs = 2
    dynamic_params_dims = 441 
    dynamic_encoder_heads = 4
        
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            
                            # (21) Adding masks weight
                            # 'frame_{}_loss_mask'.format(i): args.mask_loss_coef,
                            'frame_{}_loss_dice'.format(i): args.dice_loss_coef,
                            
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
        
    # (22) Including masks
    # losses = ['labels', 'boxes']
    losses = ['labels', 'boxes', 'masks']
    
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_seg_fcs = num_seg_fcs, 
        dynamic_params_dims = dynamic_params_dims, 
        dynamic_encoder_heads = dynamic_params_dims, 
        mask_positional_encoding_cfg=dict(type='RelSinePositionalEncoding', num_feats=4, normalize=True),
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
    )
    return model, criterion, postprocessors
