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
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, masks_to_boxes
from util.mask_ops import mask_iou_calculation, batch_dice_loss, batch_sigmoid_focal_loss
from models.structures import Instances
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 # (1) Adding mask
                #  cost_mask_dice: float = 1,
                 cost_mask: float = 1,
                 cost_dice: float = 1,
                 ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # (2) Adding masks
        # self.cost_mask_dice = cost_mask_dice
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, use_focal=True):
        
        def save_image(feature_map, layer_name):
            image_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-mask-AppleMots/output/pred_masks/matcher_py"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i in range(feature_map.size(0)):
                plt.imshow(feature_map[i, 0].detach().cpu().numpy(), cmap='gray')
                plt.title(f"{layer_name}_{i}")
                filename = f"{layer_name}_{i}_{timestamp}.png"
                plt.savefig(os.path.join(image_path, filename))
                plt.close()
                
                
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            if use_focal:
                out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            # print("targets[0] in matcher is:", targets)
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            if use_focal:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
                
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            
            # (3) Adding out_mask
            if 'pred_masks' in outputs:
                out_mask = outputs['pred_masks']  # [batch_size * num_queries, H, W]
                if isinstance(targets[0], Instances):
                    tgt_mask = torch.cat([gt_per_img.masks for gt_per_img in targets])
                else:
                    tgt_mask = torch.cat([v['masks'] for v in targets]) 
                # print('tgt_mask in matcher:', tgt_mask.shape)
                # print('out_mask in matcher:', out_mask.shape)
                # print('out_mask[:, None] in matcher:', out_mask[:, None].shape)
                
                out_mask = interpolate(out_mask, size=tgt_mask.shape[-2:],
                                mode="nearest")
                # save_image(out_mask, 'matcher')
                # print('out_mask after interpolate in matcher:', out_mask.shape)
                out_mask = out_mask.squeeze(0).flatten(1,-1)
                # print('out_mask in matcher:', out_mask.shape)
                # print('tgt_mask before in matcher:', tgt_mask.shape)
                # tgt_mask = tgt_mask[:, 0].flatten(1)
                tgt_mask = tgt_mask.flatten(1)
                # print('tgt_mask in matcher:', tgt_mask.shape)
                # num_boxes = sum(len(gt_per_img.boxes) for gt_per_img in targets) if isinstance(targets[0], Instances) else sum(len(v["boxes"]) for v in targets)
                # cost_mask_dice = mask_iou_calculation (out_mask, tgt_mask)
                
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            
                # cost_mask_focal = compute_focal_loss_matrix (out_mask, tgt_mask)
                
                # C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
                # C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_mask_dice * cost_mask_dice )
                C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_dice * cost_dice + self.cost_mask * cost_mask)
                C = C.view(bs, num_queries, -1).cpu()
            else: 
                C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
                C = C.view(bs, num_queries, -1).cpu()
            
            # C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
            # C = C.view(bs, num_queries, -1).cpu()
                
            if isinstance(targets[0], Instances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            else:
                sizes = [len(v["boxes"]) for v in targets]

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # (gt idx, pred_idx)


def build_matcher(args):
        return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            # cost_mask_dice = args.set_cost_mask_dice,
                            cost_mask = args.set_cost_mask,
                            cost_dice = args.set_cost_dice
                            )





