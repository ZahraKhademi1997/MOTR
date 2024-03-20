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

    # unique_values_mask1 = np.unique(mask1)
    # unique_values_mask2 = np.unique(mask2)
    # print("Unique values in mask1:", unique_values_mask1)
    # print("Unique values in mask2:", unique_values_mask2)
    
    # # Verify mask1 and mask2 are valid binary masks
    # if not np.all(np.logical_or(mask1 == 0, mask1 == 1)):
    #     print("Invalid binary mask detected in mask1.")
    # if not np.all(np.logical_or(mask2 == 0, mask2 == 1)):
    #     print("Invalid binary mask detected in mask2.")

    # Check for NaN values in intersection and union
    # if np.isnan(np.sum(intersection)):
    #     print("NaN values detected in the intersection array.")
    # if np.isnan(np.sum(union)):
    #     print("NaN values detected in the union array.")

    # Convert intersection and union to float
    # intersection_sum = np.sum(intersection).astype(float)
    # union_sum = np.sum(union).astype(float)

    # intersection_sum = torch.sum(intersection).detach().cpu().numpy().astype(float)
    # union_sum = torch.sum(union).detach().cpu().numpy().astype(float)
    intersection_sum = torch.sum(intersection)
    union_sum = torch.sum(union)
    
    # print('intersection has the type of:', intersection_sum.dtype)
    # print('union has the type of:', union_sum.dtype)

    # Check if intersection_sum or union_sum is zero
    # if intersection_sum == 0 or union_sum == 0:
    #     print("Intersection or union is zero. Cannot calculate IoU.")
    #     return 0.0

    iou = intersection_sum / union_sum
    # print('iou in compute_iou is:', iou)
    return iou


def mask_iou_calculation(pred_masks, gt_masks):
    # pred_masks = [mask.to('cuda:0') for mask in pred_masks]
    # gt_masks = [mask.to('cuda:0') for mask in gt_masks]
    # print('gt_mask has the shape of:',gt_masks.shape)
    # print('pred_mask has the shape of:',pred_masks.shape)
    num_pred_masks = len(pred_masks)
    num_gt_masks = len(gt_masks)

    iou_matrix = torch.zeros(num_pred_masks, num_gt_masks)

    for i in range(num_pred_masks):
        for j in range(num_gt_masks):
            iou = compute_iou(pred_masks[i], gt_masks[j])
            iou_matrix[i, j] = 1 - iou
            
    return iou_matrix