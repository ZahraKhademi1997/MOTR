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
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # print('boxes1:', boxes1.shape, 'boxes2:', boxes2)
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    if torch.isnan(boxes1).any() or torch.isnan(boxes2).any():
        raise ValueError("Input boxes contain NaN values.")

    # Check that boxes are valid (x1 < x2 and y1 < y2)
    if not (boxes1[:, 2:] > boxes1[:, :2]).all() or not (boxes2[:, 2:] > boxes2[:, :2]).all():
        raise ValueError("Boxes are not valid. Ensure that x1 < x2 and y1 < y2.")



    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

# def generalized_box_iou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#     The boxes should be in [x0, y0, x1, y1] format

#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#     """
#     # Ensure input tensors are on the same device
#     device = boxes1.device

#     # Initialize an output tensor with zeros (or another default invalid value)
#     iou_matrix = torch.zeros((boxes1.size(0), boxes2.size(0)), device=device)

#     # Identify boxes with NaNs and create masks for valid boxes
#     valid_boxes1 = ~torch.isnan(boxes1).any(dim=1)
#     valid_boxes2 = ~torch.isnan(boxes2).any(dim=1)

#     # Filter valid boxes
#     boxes1_filtered = boxes1[valid_boxes1]
#     boxes2_filtered = boxes2[valid_boxes2]

#     # Calculate intersection-over-union for valid boxes only
#     if boxes1_filtered.size(0) > 0 and boxes2_filtered.size(0) > 0:
#         iou, union = box_iou(boxes1_filtered, boxes2_filtered)

#         # Calculate the coordinates for the union of boxes
#         lt = torch.min(boxes1_filtered[:, None, :2], boxes2_filtered[:, :2])
#         rb = torch.max(boxes1_filtered[:, None, 2:], boxes2_filtered[:, 2:])

#         # Calculate area of union
#         wh = (rb - lt).clamp(min=0)  # [N_filtered,M_filtered,2]
#         area = wh[:, :, 0] * wh[:, :, 1]

#         giou = iou - (area - union) / area

#         # Place calculated GIoU into the appropriate locations in the iou_matrix
#         iou_matrix[valid_boxes1][:, valid_boxes2] = giou

#     return iou_matrix



def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float).to (masks.device)
    x = torch.arange(0, w, dtype=torch.float).to (masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
