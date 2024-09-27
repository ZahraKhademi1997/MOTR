
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
from util.box_ops import masks_to_boxes, normalize_boxes
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, MHAttentionMap, MaskHeadSmallConv, 
                            dice_loss, sigmoid_focal_loss, sigmoid_ce_loss, SpatialMLP)
from util.axial_attention import AxialAttention, AxialBlock
# from models.mmcv_utils.mask_position_encoding import build_positional_encoding
from util.cross_attention import CrossAttentionLayer
from util.PerPixelEmbedding import PerPixelEmbedding
from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import code
from datetime import datetime
import os
from util.linear import Linear
from util.points import get_uncertain_point_coords_with_randomness, point_sample
from pathlib import Path
from torch.nn.init import xavier_uniform_
import torch.nn.init as init
from util.FPN_encoder import FPNEncoder
from util.misc import inverse_sigmoid
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'



class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        num_points,
                        oversample_ratio,
                        importance_sample_ratio,
                        dn_losses=[],
                        dn=False,
                        ):
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
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        self.dn=dn
        self.dn_losses=dn_losses

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
        
        # assert all(len(gt) > 0 for gt in gt_instances), "Empty ground truth instances."
        # assert all(isinstance(ind, tuple) and len(ind) == 2 for ind in indices), "Indices format is incorrect."
        
        # for i, (src_per_img, tgt_per_img) in enumerate(indices):
        #     # print(f"Max src index: {src_per_img.max()}, Outputs size: {outputs['pred_logits'][i].shape[0]}")
        #     # print(f"Max tgt index: {tgt_per_img.max()}, GT instances size: {len(gt_instances[i])}")
        #     assert src_per_img.max() < outputs['pred_logits'][i].shape[0], "Source index out of bounds."
        #     assert tgt_per_img.max() < len(gt_instances[i]), "Target index out of bounds."
        
    
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
                                             alpha=-1,
                                             gamma=0,
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
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        # filtered_idx = []
        # for src_per_img, tgt_per_img in indices:
        #     keep = tgt_per_img != -1
        #     filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        # indices = filtered_idx
        # idx = self._get_src_permutation_idx(indices)
        # src_boxes = outputs['pred_boxes'][idx]
        # target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        # target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        # mask = (target_obj_ids != -1)
        
        
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # masks = [t["masks"] for t in targets]
        masks = []
        # print(gt_instances)
        for t in gt_instances:
            masks_field = t.get('masks')
            masks.append(masks_field)
            
        #######################################################################################
        # Plotting sampling points
        def save_sampled_points_on_masks(src_mask, tgt_mask, point_coords, index=0):
            """
            Save sampled points on both the prediction and groundtruth masks to files.
            
            Args:
            - src_mask (Tensor): The predicted masks tensor [H, W].
            - tgt_mask (Tensor): The groundtruth masks tensor [H, W].
            - point_coords (Tensor): The coordinates of sampled points [num_points, 2].
            - output_dir (str): Directory to save the images.
            - index (int): Index of the sample to plot in batch.
            """
            if src_mask.size(0) > index and tgt_mask.size(0) > index: 
                src_mask = src_mask[index].squeeze().cpu().numpy()  # Squeeze and move tensor to CPU
                tgt_mask = tgt_mask[index].squeeze().cpu().numpy()  # Squeeze and move tensor to CPU
                point_coords = point_coords[index].cpu().numpy()  # Move tensor to CPU

                # Normalize and scale point coordinates to match image dimensions
                point_coords[:, 0] *= src_mask.shape[0]  # Scale y coordinates
                point_coords[:, 1] *= src_mask.shape[1]  # Scale x coordinates

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # Plot for source mask
                axs[0].imshow(src_mask, cmap='gray')
                axs[0].scatter(point_coords[:, 1], point_coords[:, 0], color='green', s=0.5)
                axs[0].set_title('Sampled Points on Prediction Mask')

                # Plot for target mask
                axs[1].imshow(tgt_mask, cmap='gray')
                axs[1].scatter(point_coords[:, 1], point_coords[:, 0], color='green', s=0.5)
                axs[1].set_title('Sampled Points on Groundtruth Mask')
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_{timestamp}.png"
                filepath = os.path.join('/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-segmentation-head-training/output/sample_points', filename)
        
                # Save the figure
                plt.savefig(filepath)
                plt.close()
            
            
        # Check if all masks in the list are non-empty
        # def create_false_mask_with_size(size, device):
        #     # Adjust the size to ensure it has three dimensions
        #     return torch.zeros((1, size[0], size[1]), dtype=torch.bool, device=device)
        
        
        # def duplicate_first_non_empty_mask(masks):
        #     # Find the first non-empty mask
        #     non_empty_mask = None
        #     for mask in masks:
        #         if mask.nelement() > 0:
        #             non_empty_mask = mask
        #             break

        #     # If no non-empty mask is found, return None
        #     if non_empty_mask is None:
        #         return None

        #     # Create new masks, replacing empty ones with a duplicate of the first non-empty mask
        #     new_masks = []
        #     for mask in masks:
        #         if mask.nelement() == 0:
        #             new_masks.append(non_empty_mask.clone())  # Duplicate the first non-empty mask
        #         else:
        #             new_masks.append(mask)

        #     return new_masks
        
        
        # if all(mask.nelement() > 0 for mask in masks):
        #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            
        # elif any(mask.nelement() > 0 for mask in masks):
        #     # Some masks are empty, but not all. Use the duplication function.
        #     new_masks = duplicate_first_non_empty_mask(masks)
        #     if new_masks is not None:
        #         target_masks, valid = nested_tensor_from_tensor_list(new_masks).decompose()
                
        # else:
        #     new_masks = []
        #     for mask in masks:
        #         if mask.nelement() == 0:
        #             size = mask.size()[1:] if mask.size()[0] == 0 else mask.size()
        #             new_mask = create_false_mask_with_size(size, mask.device)
        #             new_masks.append(new_mask)
        #         else:
        #             new_masks.append(mask)

        #     # Decompose the new list of masks
        #     target_masks, valid = nested_tensor_from_tensor_list(new_masks).decompose()
        ########################################################################################
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        if target_masks.numel() == 0:
            print("Warning: No target indices or target masks available.")
            
        size = target_masks.shape[-2:]
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            
            # if target_masks.numel() != 0:
            #     save_sampled_points_on_masks(src_masks, target_masks, point_coords, index=0)
            
            
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        losses = {
            # "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_boxes),
            "loss_dice": dice_loss(point_logits, point_labels, size, num_boxes),
        }

        del src_masks
        del target_masks
        return losses
    
    def autoencoder_loss(self, reconstructed):
        
        # Compute MSE loss between predicted and GT boxes (reduction='sum' over individual elements)
        loss_ae = F.mse_loss(reconstructed['pred_boxes'], reconstructed['input'], reduction='mean')

        losses = {}
        losses['loss_ae'] = loss_ae 

        return losses
        
    
    # Add DN for the train
    def prep_for_dn(self,mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar,pad_size=mask_dict['scalar'],mask_dict['pad_size']
        assert pad_size % scalar==0
        single_pad=pad_size//scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes,num_tgt,single_pad,scalar
    
    
    def match_for_single_frame(self, outputs: dict):
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        
        # Retrieve the matching between the outputs of the last layer and the targets
        # if self.dn is not False and mask_dict is not None:
        #     output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
        #     exc_idx = []
            
        #     if len(gt_instances_i.labels) > 0:
        #         t = torch.arange(0, len(gt_instances_i.labels)).long().cuda()
        #         t = t.unsqueeze(0).repeat(scalar, 1)
        #         tgt_idx = t.flatten()
        #         output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
        #         output_idx = output_idx.flatten()
        #     else:
        #         output_idx = tgt_idx = torch.tensor([]).long().cuda()
        #     exc_idx.append((output_idx, tgt_idx))



        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
        pred_masks_i = track_instances.pred_masks  # predicted masks of i-th image.
        
        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
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
        # assert not torch.isnan(unmatched_outputs['pred_boxes']).any(), "NaN found in unmatched_outputs[pred_boxes] in MOTR"
        # output_dir_unmatched_outputs_bbox = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/unmatched_outputs_bbox.txt"
        # with open (output_dir_unmatched_outputs_bbox, 'w') as f:
        #     f.write (str(track_instances.pred_boxes[unmatched_track_idxes]))
            
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        # assert not torch.isnan(active_track_boxes).any(), "NaN found in active_track_boxes in MOTR"

        
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            
            # (17) Adding active track masks
            # track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

            # gt_masks_to_boxes = box_ops.box_cxcywh_to_xyxy(masks_to_boxes(gt_masks.float())).to(active_track_masks.device)
            # active_track_masks_to_boxes = box_ops.box_cxcywh_to_xyxy(masks_to_boxes(active_track_masks)).to(active_track_masks.device)
            # track_instances.iou_masks[active_idxes] = matched_boxlist_iou(Boxes(active_track_masks_to_boxes), Boxes(gt_masks_to_boxes))
       
        def plot_and_save_masks(active_idxes, predicted_masks, ground_truth_masks, active_gt_boxes, active_predicted_boxes, output_dir):
            # active_predictions = predicted_masks[active_idxes].detach().sigmoid()
            active_predictions = predicted_masks[active_idxes]
            num_active = active_predictions.shape[0]
            num_gt = ground_truth_masks.shape[0]
            num_plots = max(num_active, num_gt)
            active_gt_boxes = active_gt_boxes.detach().cpu()
            active_predicted_boxes =  active_predicted_boxes.detach().cpu()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
            # Masks and Boxes
            for i in range(num_plots):
                if i == 1:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    if i < num_active:
                        # print('active_predicted_boxes[i]:', active_predicted_boxes[i])
                        x1_p, y1_p, x2_p, y2_p = active_predicted_boxes[i]
                        mask_height_p, mask_width_p = active_predictions[i].shape[0] , active_predictions[i].shape[1]
                        # print('x1_p:', x1_p, 'y1_p:', y1_p, 'x2_p:', x2_p, '2_p:', y2_p) # x1_p: tensor(0.7680) y1_p: tensor(0.0493) x2_p: tensor(0.7896) y2_p: tensor(0.0750)
                        axes[0].imshow(active_predictions[i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
                        rect_pred = patches.Rectangle((x1_p*mask_width_p, y1_p * mask_height_p),
                                                    x2_p * mask_width_p- x1_p * mask_width_p,
                                                    y2_p * mask_height_p - y1_p * mask_height_p,
                                                    linewidth=2, edgecolor='r', facecolor='none')
                        axes[0].add_patch(rect_pred)
                        axes[0].set_title(f"Active Pred Mask {i+1}")
                        axes[0].axis('off')

                    if i < num_gt:
                        x1, y1, x2, y2 = active_gt_boxes[i]
                        mask_height, mask_width = ground_truth_masks[i].shape[0] , ground_truth_masks[i].shape[1]
                        axes[1].imshow(ground_truth_masks[i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
                        rect_gt = patches.Rectangle((x1 * mask_width, y1 * mask_height),
                                                    x2 * mask_width - x1 * mask_width,
                                                    y2 * mask_height - y1 * mask_height,
                                                    linewidth=2, edgecolor='g', facecolor='none')
                        axes[1].add_patch(rect_gt)
                        axes[1].set_title(f"GT Mask {i+1}")
                        axes[1].axis('off')

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"mask_comparison_{i}_{timestamp}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close()


        # (15) Adding active track masks
        active_track_masks_primarly = track_instances.pred_masks[active_idxes]
        if len(active_track_masks_primarly) > 0: 
            gt_masks = gt_instances_i.masks[track_instances.matched_gt_idxes[active_idxes]].float()
            plot_and_save_masks(active_idxes, track_instances.pred_masks, gt_masks, gt_boxes, active_track_boxes, '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-trk-QS/output/criterion')

        # active_track_masks = active_track_masks.sigmoid()
        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)
        
        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        
                 # Loss calculation for Hungarian between matched indices
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})
            
            
       
        # # Hungarian between DN indices
        # if self.dn != "no" and mask_dict is not None:
        #     for loss in self.losses:
        #         dn_loss = self.get_loss(loss, 
        #                                 outputs=output_known_lbs_bboxes,
        #                                 gt_instances=[gt_instances_i],
        #                                 indices=exc_idx,
        #                                 num_boxes=1*scalar)
        #         self.losses_dict.update(
        #             {'frame_{}_{}_dn'.format(self._current_frame_idx, key): value for key, value in dn_loss.items()})
            
        # elif self.dn is not False:
        #     l_dict = dict()
        #     l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to(pred_logits_i.device)
        #     l_dict['loss_giou_dn'] = torch.as_tensor(0.).to(pred_logits_i.device)
        #     l_dict['loss_ce_dn'] = torch.as_tensor(0.).to(pred_logits_i.device)
        #     if self.dn == "seg":
        #         l_dict['loss_mask_dn'] = torch.as_tensor(0.).to(pred_logits_i.device)
        #         l_dict['loss_dice_dn'] = torch.as_tensor(0.).to(pred_logits_i.device)
        #     self.losses_dict.update(
        #         {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
        #         l_dict.items()})


        # Hungarian between Aux indices
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):    
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_masks': aux_outputs['pred_masks'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    # if loss == 'masks':
                    #     # Intermediate masks losses are too costly to compute, we ignore them.
                    #     continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
                    
 
        # Hungarian between GT and prediction indices in two-stage
        if 'interm_outputs' in outputs:
            # print("Full pred_boxes shape:", outputs['interm_outputs']['pred_boxes'].shape)
            interm_outputs = outputs['interm_outputs']
            unmatched_outputs_layer_interm = {
                    'pred_logits': interm_outputs['pred_logits'],
                    'pred_boxes': interm_outputs['pred_boxes'],
                    'pred_masks': interm_outputs['pred_masks'],
                }
            interm_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer_interm, self.matcher)
            
            for loss in self.losses:
                l_dict = self.get_loss(loss,
                                       interm_outputs,
                                       gt_instances=[gt_instances_i],
                                       indices=[(interm_matched_indices_layer[:, 0], interm_matched_indices_layer[:, 1])],
                                       num_boxes=1)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                self.losses_dict.update(
                    {'frame_{}_aux_{}'.format(self._current_frame_idx, key): value for key, value in
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
        masks = torch.nn.functional.interpolate(out_mask.unsqueeze(0), size=target_size, mode='bilinear').squeeze(0)
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
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed, num_seg_fcs, mask_positional_encoding_cfg={'type': 'RelSinePositionalEncoding', 'num_feats': 128, 'normalize': True},
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False, initial_pred=True):
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
        self.num_seg_fcs = num_seg_fcs
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
        # if with_box_refine:
        #     self.class_embed = _get_clones(self.class_embed, num_pred)
        #     self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        #     nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        #     # hack implementation for iterative bounding box refinement
        #     self.transformer.decoder.bbox_embed = self.bbox_embed
        # else:
        #     nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        #     self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        #     self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        #     self.transformer.decoder.bbox_embed = None
        # if two_stage:
        #     # hack implementation for two-stage
        #     # self.transformer.decoder.class_embed = self.class_embed
        #     for box_embed in self.bbox_embed:
        #         nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        nn.init.constant_(self.transformer._bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.transformer._bbox_embed for _ in range(num_pred)]) 
        self.mask_embed = nn.ModuleList([self.transformer.mask_embed for _ in range(num_pred)]) 
        
        # segmentation head
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        hid_dim = hidden_dim * 2
        num_feature_levels = 3
        self.PerPixelEmbedding = PerPixelEmbedding(
            backbone.output_shape(), 
            hidden_dim, 
            hidden_dim,
            norm = None) # Initialized
        self.AxialBlock = AxialBlock(hidden_dim,hidden_dim // 2) # Initialized
        
        # Two-stage prediction heads
        self.position = nn.Embedding(self.num_queries, 4)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.forward_prediction_heads = self.transformer.forward_prediction_heads
        self.dn_post_process = self.transformer.dn_post_process
        self.initial_pred = initial_pred
        nn.init.uniform_(self.position.weight.data, 0, 1)
        

        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        # self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length
        self.mem_bank_len = 0 if memory_bank is None else 4
        
    def _generate_empty_tracks(self, frame_shape, device):
        track_instances = Instances((1, 1))
        num_queries, dim  = self.query_embed.weight.shape

        if self.transformer.content_det is None and self.transformer.pos_det is None:
            print("No detection embedding found, initialize with random weights")
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
            # track_instances.ref_pts = torch.zeros((num_queries, 4), device=device)
            # track_instances.query_pos = torch.zeros((num_queries, 256), device=device)
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, self.transformer.pos_det.weight])
            track_instances.query_pos = torch.cat([self.query_embed.weight, self.transformer.content_det.weight])
            
        track_instances.output_embedding = torch.zeros((len(track_instances), dim), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.pred_masks = torch.zeros((len(track_instances), frame_shape[0], frame_shape[1]), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)
        return track_instances.to(device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]
            
    
    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device
        
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.transformer.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig) 
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list
        

    def _forward_single_image(self, samples, targets, track_instances: Instances):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        
        # Per-Pixel Decoding
        embeddings, multi_scale_features = self.PerPixelEmbedding(features, (samples.tensors.shape[2], samples.tensors.shape[3]))
        attention_embedding, similarity_h, similarity_w = self.AxialBlock(embeddings)
        
        bs = features[-1].tensors.shape[0]
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        hs, init_reference, inter_references, interm_outputs = self.transformer(srcs, masks, pos, embeddings, attention_embedding, targets, track_instances.query_pos,track_instances.ref_pts, mem_bank=track_instances.mem_bank, mem_bank_pad_mask=track_instances.mem_padding_mask)
        hs = torch.stack(hs, dim=0)
        inter_references = torch.stack(inter_references, dim=0)
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
            tmp = self.bbox_embed[lvl](hs[lvl])
            dynamic_params = self.mask_embed[lvl](hs[lvl])
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

        ref_pts_all = torch.cat([init_reference[:,:,:2].unsqueeze(0), inter_references[:, :, :, :2]], dim=0)
            
        # Segmentation Head
        F_embeddings = embeddings.flatten(2).transpose(1, 2)
        F_embeddings = F_embeddings.view(F_embeddings.shape[1], F_embeddings.shape[2])
        num_imgs = outputs_class[-1].size(0)
        # for i in range(num_imgs): 
        #     pos_dynamic_params = outputs_dynamic_params[-1][i]
        #     cross_attended_output, cross_attn_map = self.trasnformer.pos_cross_attention(
        #                 tgt=pos_dynamic_params,
        #                 memory=F_embeddings,
        #             )
        #     cross_attended_output = cross_attended_output.unsqueeze(0)
        #     pred_masks = torch.einsum("bqc,bchw->bqhw", cross_attended_output, attention_embedding)
        #     pred_masks = pred_masks.sigmoid()
        for i in range(num_imgs):
            all_layer_masks = []
            for layer_dynamic_params in outputs_dynamic_params:  # Iterate over all layers
                pos_dynamic_params = layer_dynamic_params[i]
                cross_attended_output = self.transformer.pos_cross_attention(
                    tgt=pos_dynamic_params,
                    memory=F_embeddings,
                ).unsqueeze(0)
                pred_mask = torch.einsum("bqc,bchw->bqhw", cross_attended_output, attention_embedding)
                all_layer_masks.append(pred_mask.sigmoid())  # Append current layer's predictions

            # Stack all layer predictions
            pred_masks = torch.stack(all_layer_masks, dim=0)  # Shape: [num_layers, num_imgs, num_queries, H, W]
    
        # print('pred_masks:', pred_masks[-1].shape, 'outputs_class:', outputs_class.shape, 'outputs_coord:', outputs_coord.shape)   
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5], 'pred_masks': F.interpolate(
                pred_masks[-1],
                size=(samples.tensors.shape[2], samples.tensors.shape[3]),
                mode="bilinear",
                align_corners=False,
            )}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, pred_masks, outputs_coord)
        out['hs'] = hs[-1]

        out['interm_outputs'] = interm_outputs # query selection from encoder
        
        return out
    
    
    
    def _post_process_single_image(self, frame_res, track_instances, frame_shape, is_last):             
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()
        
        high_scores = track_scores[track_scores > 0.5]
        print("Track scores:", high_scores)
        
        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.pred_masks = frame_res['pred_masks'][0]
        # print('track_instances.pred_masks:', track_instances.pred_masks.shape)
        track_instances.output_embedding = frame_res['hs'][0]

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
            
        if self.memory_bank is not None:
            # print(f"Type of self.memory_bank: {type(self.memory_bank)}")
            track_instances = self.memory_bank(track_instances)
            out_dir= '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-trk-QS/output/trk_inst_mem.txt'
            with open (out_dir, 'w') as f:
                f.write(str(track_instances))
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
                
        tmp = {}
        # tmp['init_track_instances'] = self._generate_empty_tracks(((track_instances.pred_masks.shape[1], track_instances.pred_masks.shape[2])), frame_res['pred_masks'].device)
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else: 
            frame_res['track_instances'] = None
        
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(ori_img_size, 'cuda:0')
        else:
            empty = self._generate_empty_tracks(ori_img_size, 'cuda:0')
            if not hasattr(track_instances, 'pred_boxes'):
                # Assuming `track_instances` should have the same number of queries as the output of `_generate_empty_tracks`
                num_queries = len(track_instances)
                
                # Initialize pred_boxes to a default value (e.g., zeros). Adjust dimensions as needed.
                track_instances.pred_boxes = torch.zeros((num_queries, 4), device=track_instances.query_pos.device)
                track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=track_instances.query_pos.device)
                track_instances.pred_masks = torch.zeros((len(track_instances), empty.pred_masks.shape[1], empty.pred_masks.shape[2]), dtype=torch.float, device=track_instances.query_pos.device)

            track_instances = Instances.cat([
                self._generate_empty_tracks(ori_img_size, 'cuda:0'),
                track_instances])
            
        res = self._forward_single_image(img,
                                        track_instances=track_instances)
        
        res = self._post_process_single_image(res, track_instances, ori_img_size,False)
        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        
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
        
        targets = data['gt_instances']
        frames = data['imgs']  # list of Tensor.
        device = frames[0].device
        self.mask_height , self.mask_width = frames[0].shape[1] , frames[0].shape[2]
        frames_shape = (self.mask_height , self.mask_width)
        
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'pred_masks': [],   
        }

        track_instances = None
        keys = list(self._generate_empty_tracks(frames_shape, device)._fields.keys())
    
        for frame_index, (frame, target) in enumerate(zip(frames, targets)):
            frame_shape = (frame.shape[1], frame.shape[2])
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            
            
            if track_instances is None:
                track_instances = self._generate_empty_tracks(frame_shape, device)
            else:
                # print('frame_shape:', frame_shape, 'track_instances:', track_instances.pred_masks.shape, 'generated:', self._generate_empty_tracks(frame_shape, device).pred_masks.shape)
                empty_tracks = self._generate_empty_tracks(frame_shape, device)
                new_shape = empty_tracks.pred_masks.shape[-2:]
                existing_shape = track_instances.pred_masks.shape[-2:]
                if new_shape != existing_shape:
                    empty_tracks.pred_masks = F.interpolate(
                        empty_tracks.pred_masks.unsqueeze(0),
                        size=existing_shape,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                        
                track_instances = Instances.cat([empty_tracks, track_instances])
                
                
            if self.use_checkpoint and frame_index < len(frames) - 2:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['pred_masks'],
                        frame_res['ref_pts'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        # *[aux['pred_masks'] for aux in frame_res['aux_outputs']]
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
                frame_res = self._forward_single_image(frame, target, track_instances)
            frame_res = self._post_process_single_image(frame_res, track_instances, frame_shape,is_last)
        
            track_instances = frame_res['track_instances']

            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
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
        'e2e_joint': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)
    num_seg_fcs = 2
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
                            # 'frame_{}_loss_mask'.format(i): args.mask_loss_coef,
                            'frame_{}_loss_dice'.format(i): args.dice_loss_coef,
                            
                            })
            

    # Weights for auxiliary and intermediate output losses
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    # 'frame_{}_aux{}_loss_mask'.format(i, j): args.mask_loss_coef,
                                    'frame_{}_aux{}_loss_dice'.format(i, j): args.dice_loss_coef,
                                    })
                    
    for i in range(num_frames_per_batch):     
        # Intermediate output losses (if applicable)
        weight_dict.update({
            'frame_{}_aux_loss_ce_interm'.format(i): args.cls_loss_coef,
            'frame_{}_aux_loss_bbox_interm'.format(i): args.bbox_loss_coef,
            'frame_{}_aux_loss_giou_interm'.format(i): args.giou_loss_coef,
            # 'frame_{}_aux_loss_mask_interm'.format(i): args.mask_loss_coef,
            'frame_{}_aux_loss_dice_interm'.format(i): args.dice_loss_coef,
            
            })


    # Optional: Memory bank weights if applicable
    if args.memory_bank_type is not None and args.memory_bank_type == 'MemoryBank':
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        # print('memory_bank:', memory_bank)
        for i in range(num_frames_per_batch):
            weight_dict.update({
                f"frame_{i}_track_loss_ce": args.cls_loss_coef  # Tracking specific class loss if using memory bank
            })
    else:
        memory_bank = None

        
    # (22) Including masks
    # losses = ['labels', 'boxes']
    losses = ['labels', 'boxes', 'masks']
    importance_sample_ratio = 0.75
    oversample_ratio = 3.0
    num_points = 12544
    
    dn_losses = []
    dn = False
    initial_pred = False

    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses, num_points = num_points, oversample_ratio = oversample_ratio, importance_sample_ratio = importance_sample_ratio, dn_losses = dn_losses, dn = dn)
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
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        initial_pred =initial_pred,
    )
    return model, criterion, postprocessors
