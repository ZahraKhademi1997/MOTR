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
from .segmentation import (DETRsegm, dice_loss, sigmoid_focal_loss)
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, masks_to_boxes
from models.structures import Instances
import numpy as np
import torch.nn.functional as F


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
                 ########################
                 # (1) Adding mask loss
                 cost_giou_mask_to_box: float = 1,
                 cost_mask_dice: float = 1,
                 cost_mask_focal: float = 1):
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
        #######################################
        # (2) Adding masks
        self.cost_giou_mask_to_box = cost_giou_mask_to_box
        self.cost_mask_dice = cost_mask_dice
        self.cost_mask_focal = cost_mask_focal
        #######################################
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, use_focal=True):
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
            # print("Number of predicted boxes (out_bbox):", out_bbox.shape[0]) #Number of predicted boxes (out_bbox): 300
            
            # Also concat the target labels and boxes
            # print("targets[0] in matcher is:", targets)
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])
            # print("Number of target boxes (tgt_bbox):", tgt_bbox.shape[0]) #Number of target boxes (tgt_bbox): 23
            
            
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
            # print('cost_boxes are:', cost_bbox)
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            # print('cost_giou has the shape of:', cost_giou.shape) #torch.Size([300, 23])
            
            ######################################################################################
            # (3) Adding out_mask
            if 'pred_masks' in outputs:
                out_mask = outputs['pred_masks'].flatten(0, 1)  # [batch_size * num_queries, H, W]
                if isinstance(targets[0], Instances):
                    tgt_mask = torch.cat([gt_per_img.masks for gt_per_img in targets])
                else:
                    tgt_mask = torch.cat([v['masks'] for v in targets]) 
                
                num_boxes = sum(len(gt_per_img.boxes) for gt_per_img in targets) if isinstance(targets[0], Instances) else sum(len(v["boxes"]) for v in targets)
            
                out_mask_to_box = masks_to_boxes(out_mask)
                tgt_mask_to_box = masks_to_boxes(tgt_mask)
                cost_giou_mask_to_box = -generalized_box_iou(box_cxcywh_to_xyxy(out_mask_to_box),
                                                box_cxcywh_to_xyxy(tgt_mask_to_box)).to(outputs["pred_logits"].device)

                # Function to calculate Dice coefficient
                # def dice_coefficient(pred, target):
                #     smooth = 1e-5
                #     pred = pred.sigmoid()
                #     intersection = (pred * target).sum()
                #     union = pred.sum() + target.sum() + smooth
                #     dice = (2. * intersection + smooth) / union
                #     return dice
                
                # def compute_dice_loss_matrix(pred_masks, tgt_masks):
                #     # print('pred_masks are:', pred_masks)
                #     num_pred = pred_masks.shape[0]  # Number of predicted masks
                #     num_tgt = tgt_masks.shape[0]    # Number of target masks
                #     dice_loss_rows = []

                #     for pred_idx in range(num_pred):
                #         dice_loss_row = []
                #         # print('pred_masks[pred_idx] is:', pred_masks[pred_idx])
                #         for tgt_idx in range(num_tgt):
                #             # Compute Dice coefficient for each pair of pred and tgt mask
                #             dice_score = dice_coefficient(pred_masks[pred_idx], tgt_masks[tgt_idx])
                #             # Compute Dice loss and append to the current row
                #             dice_loss_row.append(1 - dice_score)
                #         # Convert the row to a tensor and add it to the list
                #         dice_loss_rows.append(torch.tensor(dice_loss_row, device=pred_masks.device))

                #     # Stack all rows to create the final matrix
                #     dice_loss_matrix = torch.stack(dice_loss_rows)
                #     return dice_loss_matrix
                
                def pairwise_dice_loss(pred_masks, target_masks):
                    """
                    Compute pairwise Dice loss for a batch of predicted and target masks.
                    Args:
                        pred_masks (Tensor): Predicted masks of shape [num_pred, H, W].
                        target_masks (Tensor): Target masks of shape [num_target, H, W].
                    Returns:
                        Tensor: Pairwise Dice loss of shape [num_pred, num_target].
                    """
                    # print('pred_mask has the shape of:', pred_masks.shape)
                    # print('target_mask has the shape of:', target_masks.shape)
                    num_pred = pred_masks.shape[0]  # 300 in your case
                    num_target = target_masks.shape[0]  # 23 in your case
                    dice_loss_matrix = torch.zeros(num_pred, num_target, device=pred_masks.device)
                    pred_masks = pred_masks.sigmoid()
                    for i in range(num_pred):
                        for j in range(num_target):
                            intersection = (pred_masks[i] * target_masks[j]).sum()
                            union = pred_masks[i].sum() + target_masks[j].sum()
                            dice_score = (2 * intersection + 1e-5) / (union + 1e-5)
                            dice_loss_matrix[i, j] = 1 - dice_score
                    
                    return dice_loss_matrix
                
                
                
                def pairwise_focal_loss(pred_masks, target_masks, alpha=0.25, gamma=2.0):
                    """
                    Compute pairwise focal loss for a batch of predicted and target masks.
                    Args:
                        pred_masks (Tensor): Predicted masks of shape [num_pred, H, W].
                        target_masks (Tensor): Target masks of shape [num_target, H, W].
                        alpha (float): Alpha parameter for focal loss.
                        gamma (float): Gamma parameter for focal loss.
                    Returns:
                        Tensor: Pairwise focal loss of shape [num_pred, num_target].
                    """
                    # print('pred_mask in focal has the shape of:', pred_masks.shape)
                    # print('target_mask in focal has the shape of:', target_masks.shape)
                    num_pred = pred_masks.shape[0]  # 300 in your case
                    num_target = target_masks.shape[0]  # 23 in your case
                    focal_loss_matrix = torch.zeros(num_pred, num_target, device=pred_masks.device)
                    print(pred_masks.sigmoid())
                    for i in range(num_pred):
                        for j in range(num_target):
                            pred = pred_masks[i].sigmoid()
                            target = target_masks[j]
                            ce_loss = F.binary_cross_entropy(pred, target.float(), reduction="none")
                            p_t = pred * target + (1 - pred) * (1 - target)
                            modulating_factor = (1 - p_t) ** gamma
                            alpha_factor = alpha * target + (1 - alpha) * (1 - target)
                            focal_loss = ce_loss * modulating_factor * alpha_factor
                            focal_loss_matrix[i, j] = focal_loss.mean()
                    
                    return focal_loss_matrix



                # Example usage with debugging
                cost_mask_dice = pairwise_dice_loss(out_mask, tgt_mask).to(outputs["pred_logits"].device)
                cost_mask_focal = pairwise_focal_loss(out_mask, tgt_mask).to(outputs["pred_logits"].device)
                print("cost_mask_dice is:", cost_mask_dice)
                print("cost_mask_focal is:", cost_mask_focal)
                # print('cost_mask is:', cost_mask)
                C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_giou_mask_to_box * cost_giou_mask_to_box + self.cost_mask_dice * cost_mask_dice + self.cost_mask_focal * cost_mask_focal)
                C = C.view(bs, num_queries, -1).cpu()
                
            else:
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                C = C.view(bs, num_queries, -1).cpu()
            ######################################################################################

            
            if isinstance(targets[0], Instances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            else:
                sizes = [len(v["boxes"]) for v in targets]

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            ###########################
                            # (4) Add mask giou
                            cost_giou_mask_to_box=args.set_cost_giou_mask_to_box,
                            cost_mask_dice=args.set_cost_mask_dice,
                            cost_mask_focal=args.set_cost_mask_focal)
