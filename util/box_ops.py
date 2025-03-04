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
    assert not torch.isnan(x).any(), "NaN values detected in x in box_cxcywh_to_xyxy."
    x_c, y_c, w, h = x.unbind(-1)
    # Check for zero or negative values in width and height
    assert (w > 0).all(), "Width values must be positive and non-zero."
    assert (h > 0).all(), "Height values must be positive and non-zero."
    
    # Ensure width and height are not zero to avoid division by zero
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    # Convert the list to a tensor
    result = torch.stack(b, dim=-1)

    # Final check to ensure no NaN values are generated during the computation
    assert not torch.isnan(result).any(), "NaN values generated during conversion."

    return result

def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x), (y),
         (x + 0.5 * w), (y + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def clamp_boxes(boxes, image_width, image_height):
    """Clamps a bounding box to the image boundaries.

    Args:
        bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
        image_width: The width of the image.
        image_height: The height of the image.

    Returns:
        A clamped bounding box tuple.
    """
    output = []
    for i in range(boxes.shape[0]):
        boxes_xyxy = box_cxcywh_to_xyxy(boxes[i].clone())
        x_min, y_min, x_max, y_max = boxes_xyxy
        clamped_boxes = torch.tensor([
            max(0, x_min),
            max(0, y_min),
            min(image_width - 1, x_max),
            min(image_height - 1, y_max)
        ])
        # clamped_boxes[3] = max(clamped_boxes[3], clamped_boxes[1] + 1)
        output.append(box_xyxy_to_cxcywh(clamped_boxes))

    return torch.stack(output, dim=0)


def clamp_batch_boxes(bboxes, img_width, img_height):
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
    # valid = (boxes1[:, 2:] >= boxes1[:, :2]).all(dim=1)
    # if not valid.all():
    #     invalid_boxes = boxes1[~valid]
    #     print("Invalid boxes:", invalid_boxes)
    #     raise ValueError(f"Boxes are not valid. Ensure that x1 < x2 and y1 < y2. Found invalid boxes: {invalid_boxes}")
    # print ("boxes1:", boxes1)
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    # Check for boxes1
    assert not torch.isnan(boxes1).any(), "NaN values detected in boxes1 in generalized_box_iou."
    assert not torch.isnan(boxes2).any(), "NaN values detected in boxes2 in generalized_box_iou."
    
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        invalid_boxes1 = boxes1[~(boxes1[:, 2:] >= boxes1[:, :2]).all(axis=1)]
        raise AssertionError(f"Assertion failed: Expected all bounding boxes to have min coords <= max coords. Invalid boxes1:\n{invalid_boxes1}")

    # Check for boxes2
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        invalid_boxes2 = boxes2[~(boxes2[:, 2:] >= boxes2[:, :2]).all(axis=1)]
        raise AssertionError(f"Assertion failed: Expected all bounding boxes to have min coords <= max coords. Invalid boxes2:\n{invalid_boxes2}")


    
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

    return iou - ((area - union)+1e-6) / (area+1e-6)


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