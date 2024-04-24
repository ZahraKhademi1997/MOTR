import torch
import numpy as np 
import pandas as pd 

def compute_iou(mask1, mask2):
    # print('pred_masks type in iou:', type(mask1))
    # print('gt_masks type in iou:', type(mask2))
    
    mask1 = torch.tensor(mask1, dtype=torch.bool)
    mask2 = torch.tensor(mask2, dtype=torch.bool)
    
    # mask1 = mask1.to('cuda') if isinstance(mask1, torch.Tensor) else torch.tensor(mask1, dtype=torch.bool, device='cuda')
    # mask2 = mask2.to('cuda') if isinstance(mask2, torch.Tensor) else torch.tensor(mask2, dtype=torch.bool, device='cuda')

    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)

    # unique_values_mask1 = torch.unique(mask1)
    # unique_values_mask2 = torch.unique(mask2)
    # print("Unique values in mask1:", unique_values_mask1)
    # print("Unique values in mask2:", unique_values_mask2)

    intersection_sum = torch.sum(intersection).to(mask1.device)
    union_sum = torch.sum(union).to(mask1.device)
    
    # print('intersection has the type of:', intersection_sum.dtype)
    # print('union has the type of:', union_sum.dtype)

    # Check if intersection_sum or union_sum is zero
    # if intersection_sum == 0 or union_sum == 0:
    #     print("Intersection or union is zero. Cannot calculate IoU.")
    #     return 0.0

    iou = (2 * intersection_sum / union_sum).to(mask1.device)
    return iou


def mask_iou_calculation(pred_masks, gt_masks):
    num_pred_masks = len(pred_masks)
    num_gt_masks = len(gt_masks)

    iou_matrix = torch.zeros(num_pred_masks, num_gt_masks).to(pred_masks.device)

    for i in range(num_pred_masks):
        for j in range(num_gt_masks):
            iou = compute_iou(pred_masks[i], gt_masks[j])
            iou_matrix[i, j] = 1 - (iou)
            
    return iou_matrix