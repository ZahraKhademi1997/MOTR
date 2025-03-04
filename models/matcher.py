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
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, masks_to_boxes, box_xywh_to_xyxy
from util.mask_ops import mask_iou_calculation, batch_dice_loss, batch_sigmoid_ce_loss
from models.structures import Instances
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from util.points import get_uncertain_point_coords_with_randomness, point_sample

    
batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule




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
        self.num_points = 12544
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, use_focal=True):
        
        def save_image(feature_map, layer_name):
            image_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-AppleMOTS-Axial_Cross_Attention/output/matcher"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i in range(feature_map.size(0)):
                plt.imshow(feature_map[i, 0].detach().cpu().numpy(), cmap='gray')
                plt.title(f"{layer_name}_{i}")
                filename = f"{layer_name}_{i}_{timestamp}.png"
                plt.savefig(os.path.join(image_path, filename))
                plt.close()
                
        def safe_linear_sum_assignment(cost_matrix):
            # Ensure the cost_matrix is a numpy array
            if isinstance(cost_matrix, torch.Tensor):
                cost_matrix = cost_matrix.cpu().numpy()

            if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
                print("Warning: Cost matrix contains NaN or Inf values. Replacing them with a large number.")
                # Find the maximum value among valid (finite) entries
                valid_entries = np.isfinite(cost_matrix)
                max_valid = np.max(cost_matrix[valid_entries]) if np.any(valid_entries) else 1
            
                # Replace NaN or Inf with a very large number to effectively exclude them from the assignment
                cost_matrix[~valid_entries] = max_valid + 1e5
                


            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return row_ind, col_ind  
        
        def clamp_boxes(bboxes, img_width, img_height):
            if not isinstance(bboxes, torch.Tensor):
                bboxes = torch.tensor(bboxes, dtype=torch.float32)

            # Apply max and min in a list comprehension for each bounding box coordinate
            clamped_x1 = torch.tensor([max(0, x) for x in bboxes[:, 0]], dtype=torch.float32)
            clamped_y1 = torch.tensor([max(0, y) for y in bboxes[:, 1]], dtype=torch.float32)
            clamped_x2 = torch.tensor([min(img_width - 1, x) for x in bboxes[:, 2]], dtype=torch.float32)
            clamped_y2 = torch.tensor([min(img_height - 1, y) for y in bboxes[:, 3]], dtype=torch.float32)

            # Stack the clamped coordinates into a single tensor
            clamped_boxes = torch.stack((clamped_x1, clamped_y1, clamped_x2, clamped_y2), dim=1).to(bboxes.device)

            return clamped_boxes
        
          
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
            if "pred_logits" in outputs:
                bs, num_queries = outputs["pred_logits"].shape[:2]
            else:
                bs, num_queries = outputs["pred_boxes"].shape[:2]
                # assert not num_queries !=100, f"Error: num_queries should not be 100, but got {num_queries}"

            # We flatten to compute the cost matrices in a batch
            if "pred_logits" in outputs:
                if use_focal:
                    out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
                else:
                    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            assert not torch.isnan(out_bbox).any(), "NaN values detected in out_bbox in matcher."

            # Also concat the target labels and boxes
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            if "pred_logits" in outputs:
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

            # output_dir_src = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/out_bbox.txt"
            # with open (output_dir_src, 'w') as f:
            #     f.write (str(out_bbox))
                
            # output_dir_src_convert = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/out_bbox_convert.txt"
            # with open (output_dir_src_convert, 'w') as f:
            #     f.write (str(box_cxcywh_to_xyxy(out_bbox)))
                
            # output_dir_tgt = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/tgt_bbox.txt"
            # with open (output_dir_tgt, 'w') as f:
            #     f.write (str(tgt_bbox))
            
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
            # cost_giou = -generalized_box_iou(clamp_boxes(box_cxcywh_to_xyxy(out_bbox), outputs["pred_masks"].shape[3],outputs["pred_masks"].shape[2]),
            #                                  clamp_boxes(box_cxcywh_to_xyxy(tgt_bbox), outputs["pred_masks"].shape[3],outputs["pred_masks"].shape[2]))
            
            
            # (3) Adding out_mask
            if 'pred_masks' in outputs:
                out_mask = outputs['pred_masks'].squeeze(0)  # [batch_size * num_queries, H, W]
                if isinstance(targets[0], Instances):
                    tgt_mask = torch.cat([gt_per_img.masks for gt_per_img in targets])
                else:
                    tgt_mask = torch.cat([v['masks'] for v in targets]) 
                # print('tgt_masks:', tgt_mask.shape, 'out_mask:', out_mask.shape)
                # out_mask = interpolate(out_mask, size=tgt_mask.shape[-2:],
                #                 mode="nearest")
                # # save_image(out_mask, 'matcher')
                # out_mask = out_mask.squeeze(0).flatten(1,-1)
                # tgt_mask = tgt_mask.flatten(1)
                # cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)
                # cost_dice = batch_dice_loss(out_mask, tgt_mask)
                
                out_mask = out_mask[:, None].float()
                tgt_mask = tgt_mask[:, None].float()
                # print('tgt_masks after:', tgt_mask.shape, 'out_mask after:', out_mask.shape)
                
                point_coords = torch.rand(1, self.num_points, 2, device=tgt_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # If there's no annotations
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
         
                # print("out_bbox.device:", tgt_mask.device)
                if cost_bbox.shape[0] != cost_class.shape[0]:
                    print("Mismatch in the size of cost tensors:b/c")
                if cost_bbox.shape[0] != cost_mask.shape[0]:
                    print("Mismatch in the size of cost tensors:b/m")
                if cost_bbox.shape[0] != cost_dice.shape[0]:
                    print("Mismatch in the size of cost tensors: b/d")
                if cost_bbox.shape[0] != cost_giou.shape[0]:
                    print("Mismatch in the size of cost tensors: b/g") #Causes the problem
                
                C = (self.cost_bbox * cost_bbox.to(tgt_mask.device) + self.cost_class * cost_class.to(tgt_mask.device) + self.cost_giou * cost_giou.to(tgt_mask.device) + self.cost_dice * cost_dice.to(tgt_mask.device) + self.cost_mask * cost_mask.to(tgt_mask.device))
                # C = C.view(bs, num_queries, -1).cpu()
                
            if "pred_logits" in outputs:
                C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
                # C = C.view(bs, num_queries, -1).cpu()
                
            elif "pred_logits" not in outputs:
                C = (self.cost_bbox * cost_bbox + self.cost_giou * cost_giou)
                # C = C.view(bs, num_queries, -1).cpu()
            
            
            C = C.view(bs, num_queries, -1).cpu()   
            if isinstance(targets[0], Instances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            else:
                sizes = [len(v["boxes"]) for v in targets]
            
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            indices = [safe_linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # (gt idx, pred_idx)


def build_matcher(args):
        return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            cost_mask = args.set_cost_mask,
                            cost_dice = args.set_cost_dice
                            )





