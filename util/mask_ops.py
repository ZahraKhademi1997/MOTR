import torch
import numpy as np 
import pandas as pd 
import torch.nn.functional as F

def compute_iou(mask1, mask2):
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
    
    # Check if intersection_sum or union_sum is zero
    # if intersection_sum == 0 or union_sum == 0:
    #     print("Intersection or union is zero. Cannot calculate IoU.")
    #     return 0.0

    iou = (2 * intersection_sum / union_sum).to(mask1.device)

    return iou


def mask_iou_calculation(pred_masks, gt_masks):
    # pred_masks = [mask.to('cuda:0') for mask in pred_masks]
    # gt_masks = [mask.to('cuda:0') for mask in gt_masks]
    num_pred_masks = len(pred_masks)
    num_gt_masks = len(gt_masks)

    iou_matrix = torch.zeros(num_pred_masks, num_gt_masks).to(pred_masks.device)

    for i in range(num_pred_masks):
        for j in range(num_gt_masks):
            iou = compute_iou(pred_masks[i], gt_masks[j])
            iou_matrix[i, j] = 1 - (iou)
            
    return iou_matrix



# def batch_dice_loss(inputs, targets):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """
#     if targets.dtype != torch.float32:
#         targets = targets.float()

#     if inputs.dtype != torch.float32:
#         inputs = inputs.float()

#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
#     denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss


# def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#     Returns:
#         Loss tensor
#     """

#     if targets.dtype != torch.float32:
#         targets = targets.float()

#     if inputs.dtype != torch.float32:
#         inputs = inputs.float()
        
#     hw = inputs.shape[1]

#     prob = inputs.sigmoid()
#     focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
#         inputs, torch.ones_like(inputs), reduction="none"
#     )
#     focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
#         inputs, torch.zeros_like(inputs), reduction="none"
#     )
#     if alpha >= 0:
#         focal_pos = focal_pos * alpha
#         focal_neg = focal_neg * (1 - alpha)

#     loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
#         "nc,mc->nm", focal_neg, (1 - targets)
#     )

#     return loss / hw

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / (hw+1e-8)