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
    # output_dir_boxes1 = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/boxes1.txt"
    # with open (output_dir_boxes1, 'w') as f:
    #     f.write (str(boxes1))
        
    # output_dir_boxes2 = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/boxes2.txt"
    # with open (output_dir_boxes2, 'w') as f:
    #     f.write (str(boxes2))
        
    # boxes1 = torch.clamp(boxes1, 0, 1)
    # output_dir_boxes1_after = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/boxes1_after.txt"
    # with open (output_dir_boxes1_after, 'w') as f:
    #     f.write (str(boxes1))
        
    # Ensure boxes are valid
    valid = (boxes1[:, 2:] >= boxes1[:, :2]).all(dim=1)
    if not valid.all():
        invalid_boxes = boxes1[~valid]
        print("Invalid boxes:", invalid_boxes)
        raise ValueError(f"Boxes are not valid. Ensure that x1 < x2 and y1 < y2. Found invalid boxes: {invalid_boxes}")
    
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # if torch.isnan(boxes1).any() or torch.isnan(boxes2).any():
    #     raise ValueError("Input boxes contain NaN values.")

    # Check that boxes are valid (x1 < x2 and y1 < y2)
    # if not (boxes1[:, 2:] > boxes1[:, :2]).all() or not (boxes2[:, 2:] > boxes2[:, :2]).all():
    #     raise ValueError("Boxes are not valid. Ensure that x1 < x2 and y1 < y2.")



    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


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


def infer_bbox_format(bboxes):
    # Assuming bboxes is of shape [1, N, 4] where N is the number of boxes
    bbox_sample = bboxes[0]  # Take the first batch

    # Checking if the third value is generally greater than the first (similarly for fourth vs second)
    if (bbox_sample[:, 2] > bbox_sample[:, 0]).all() and (bbox_sample[:, 3] > bbox_sample[:, 1]).all():
        return "xyxy"
    else:
        return "xywh"
    
def normalize_boxes(bboxes, img_width, img_height):
    # Assuming bboxes is of shape [1, N, 4] and represents [cx, cy, w, h]
    # Normalize cx and w by img_width
    bboxes[:, :, 0] /= img_width
    bboxes[:, :, 2] /= img_width

    # Normalize cy and h by img_height
    bboxes[:, :, 1] /= img_height
    bboxes[:, :, 3] /= img_height
    bboxes = bboxes.clamp(0, 1)
    return bboxes