# from typing import Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import griddata
# from datetime import datetime
# import os

# def draw_and_save_images(masks, boxes, expanded_boxes, output_dir = '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_debugging/output/box_vis'):
#     # print('masks.shape:', masks.shape, 'boxes:', boxes.shape, 'expanded_boxes:', expanded_boxes.shape)
#     for i, (mask, box, exp_box) in enumerate(zip(masks, boxes, expanded_boxes)):
#         # Ensure tensors are on CPU and convert to numpy for plotting
#         mask_array = mask.cpu().numpy().squeeze()
#         box = box.cpu().numpy()
#         exp_box = exp_box.cpu().numpy().squeeze()
#         # min_box = min_box.cpu().numpy().squeeze()
#         # print('mask_array.shape:', mask_array.shape, 'box:', box.shape, 'exp_box:', exp_box.shape)
        

#         # Setup plot
#         fig, ax = plt.subplots()
#         ax.imshow(mask_array, cmap='gray')
        
#         # Draw original bounding box - Red
#         rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
#         ax.add_patch(rect)

#         # Draw expanded bounding box - Blue, with dashed style
#         rect_exp = plt.Rectangle((exp_box[0], exp_box[1]), exp_box[2] - exp_box[0], exp_box[3] - exp_box[1], linewidth=2, edgecolor='b', linestyle='--', facecolor='none')
#         ax.add_patch(rect_exp)
        
#         # Draw expanded bounding box - Blue, with dashed style
#         # rect_min = plt.Rectangle((min_box[0],min_box[1]), min_box[2] - min_box[0], min_box[3] - min_box[1], linewidth=2, edgecolor='r', linestyle='--', facecolor='none')
#         # ax.add_patch(rect_min)
        
#         # Set title and remove axes
#         ax.set_title('Mask with Bounding Boxes')
#         ax.axis('off')

#         # Save the figure
#         timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#         filename = f"{output_dir}/mask_with_boxes_{i}_{timestamp}.png"
#         plt.savefig(filename)
#         plt.close()


# def calculate_bbox_from_mask(mask):
    
#     pos = torch.where(mask)
#     if pos[0].numel() == 0 or pos[1].numel() == 0:
#         # Return an empty tensor with the correct shape and device
#         return torch.tensor([], dtype=torch.float32, device=mask.device)
    
#     # Use torch.any() to find rows and columns that contain at least one True value
#     rows = torch.any(mask, dim=1)
#     cols = torch.any(mask, dim=0)
#     if not rows.any() or not cols.any():
#         return None  # Handle cases where there are no objects

#     ymin, ymax = torch.where(rows)[0][[0, -1]]
#     xmin, xmax = torch.where(cols)[0][[0, -1]]
#     # return torch.tensor([xmin*mask.shape[0], ymin*mask.shape[1], xmax*mask.shape[0], ymax*mask.shape[1]], device=mask.device)
#     return torch.tensor([xmin, ymin, xmax, ymax], device=mask.device)


# # def add_margin_to_boxes(boxes, margin, image_width, image_height):
# #     """Add margins to boxes to create an expanded and a contracted box, ensuring they don't go beyond image boundaries."""
# #     margin_max, margin_min = margin
# #     # Clone original boxes to create new sets for modification
# #     new_boxes_max = boxes.clone()
# #     new_boxes_min = boxes.clone()

# #     # Applying maximum margin to expand the boxes
# #     new_boxes_max[:, 0] = torch.clamp(new_boxes_max[:, 0] - margin_max, min=0)  # x_min
# #     new_boxes_max[:, 1] = torch.clamp(new_boxes_max[:, 1] - margin_max, min=0)  # y_min
# #     new_boxes_max[:, 2] = torch.clamp(new_boxes_max[:, 2] + margin_max, max=image_width - 1)  # x_max
# #     new_boxes_max[:, 3] = torch.clamp(new_boxes_max[:, 3] + margin_max, max=image_height - 1)  # y_max

# #     # Applying minimum margin to contract the boxes
# #     new_boxes_min[:, 0] = torch.clamp(new_boxes_min[:, 0] + margin_min, min=0)  # x_min
# #     new_boxes_min[:, 1] = torch.clamp(new_boxes_min[:, 1] + margin_min, min=0)  # y_min
# #     new_boxes_min[:, 2] = torch.clamp(new_boxes_min[:, 2] - margin_min, max=image_width - 1)  # x_max
# #     new_boxes_min[:, 3] = torch.clamp(new_boxes_min[:, 3] - margin_min, max=image_height - 1)  # y_max

# #     # Ensure the minimum values are still less than the maximum values
# #     new_boxes_max[:, 2] = torch.max(new_boxes_max[:, 2], new_boxes_max[:, 0] + 1)
# #     new_boxes_max[:, 3] = torch.max(new_boxes_max[:, 3], new_boxes_max[:, 1] + 1)
# #     new_boxes_min[:, 2] = torch.max(new_boxes_min[:, 2], new_boxes_min[:, 0] + 1)
# #     new_boxes_min[:, 3] = torch.max(new_boxes_min[:, 3], new_boxes_min[:, 1] + 1)

# #     return new_boxes_max, new_boxes_min

# def add_margin_to_boxes(boxes, margin, image_width, image_height):
#     """Add margins to boxes to create an expanded and a contracted box, ensuring they don't go beyond image boundaries."""
#     margin_max = margin
#     # Clone original boxes to create new sets for modification
#     new_boxes_max = boxes.clone()

#     # Applying maximum margin to expand the boxes
#     new_boxes_max[:, 0] = torch.clamp(new_boxes_max[:, 0] - margin_max, min=0)  # x_min
#     new_boxes_max[:, 1] = torch.clamp(new_boxes_max[:, 1] - margin_max, min=0)  # y_min
#     new_boxes_max[:, 2] = torch.clamp(new_boxes_max[:, 2] + margin_max, max=image_width - 1)  # x_max
#     new_boxes_max[:, 3] = torch.clamp(new_boxes_max[:, 3] + margin_max, max=image_height - 1)  # y_max


#     # Ensure the minimum values are still less than the maximum values
#     new_boxes_max[:, 2] = torch.max(new_boxes_max[:, 2], new_boxes_max[:, 0] + 1)
#     new_boxes_max[:, 3] = torch.max(new_boxes_max[:, 3], new_boxes_max[:, 1] + 1)

#     return new_boxes_max

# def sample_points_from_boxes(boxes, num_points, image_width, image_height):
#     """Sample points from expanded boxes ensuring valid ranges."""
#     batch_size = boxes.shape[0]
#     points = []

#     for i in range(batch_size):
#         x_min, y_min, x_max, y_max = boxes[i, :]

#         # Ensuring min and max are correctly ordered and valid
#         x_min, x_max = min(x_min, x_max), max(x_min, x_max)
#         y_min, y_max = min(y_min, y_max), max(y_min, y_max)

#         # Sampling points within the box
#         x_coords = torch.randint(low=int(x_min), high=int(x_max), size=(num_points,))
#         y_coords = torch.randint(low=int(y_min), high=int(y_max), size=(num_points,))

#         points.append(torch.stack((x_coords.float() / image_width, y_coords.float() / image_height), dim=1))
    
#     return torch.stack(points, dim=0)

# def process_masks_and_boxes(mask_preds, mask_gt, margin, num_points):
#     mask_gt_orig= mask_gt
#     batch_size, _, height, width = mask_preds.shape
#     sampled_points = []
#     mask = mask_gt.squeeze(1)
#     # print('gt_unique:', torch.unique(mask))
    
#     # print('mask_gt.shape:', mask_gt.shape, 'mask.shape:', mask.shape)
#     # all_boxes = []
#     # all_expanded_boxes = []
#     # all_minimized_boxes = []
#     for i in range(batch_size):
#         mask_gt = mask[i]  # Assuming mask_preds is [batch_size, channels, height, width]
#         # print('mask_gt.shape after:', mask_gt.shape)
#         box_pred = calculate_bbox_from_mask(mask_preds.squeeze(1)[i] > 0.55)
        
#         if box_pred.nelement() == 0:
#             # No valid box, sample points randomly across the entire mask
#             points = torch.rand(1,num_points, 2) * torch.tensor([width, height])
#         else:
#             # Add margin to the box
#             expanded_box = add_margin_to_boxes(box_pred.unsqueeze(0),margin, width, height)

#             # Sample points from the expanded box
#             points = sample_points_from_boxes(expanded_box, num_points, width, height)
#             # print('points.shape:', points.shape)
            
#         sampled_points.append(points)
#         # all_boxes.append(box_pred)
#         # all_expanded_boxes.append(expanded_box)
#         # all_minimized_boxes.append(minimized_box)
#     # draw_and_save_images(mask_preds, all_boxes, all_expanded_boxes)
#     if sampled_points:
#         sampled_points = torch.cat(sampled_points, dim=0)
#     else:
#         sampled_points = torch.empty((0, num_points, 2), device=mask_preds.device)

#     return sampled_points.to(mask_preds.device)



# def calculate_uncertainty(sem_seg_logits):
#     # Check if there are at least 2 elements along dimension 1
#     if sem_seg_logits.size(1) < 2:
#         # Handle special case where top-2 cannot be applied
#         # For instance, return the absolute value if only 1 class exists
#         top2_scores = torch.abs(sem_seg_logits)  # Or any suitable operation
#     else:
#         # Calculate top-2 scores as intended
#         top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    
#     # Assuming uncertainty is based on the difference in scores
#     uncertainty = top2_scores[:, 0] - top2_scores[:, 1] if sem_seg_logits.size(1) >= 2 else top2_scores.squeeze(1)
#     return uncertainty


# def calculate_gradients(mask_preds):
#     # Assuming mask_preds is a tensor of shape (N, C, H, W)
#     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask_preds.device).view(1, 1, 3, 3)
#     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask_preds.device).view(1, 1, 3, 3)
    
#     grad_x = F.conv2d(mask_preds, sobel_x, padding=1, groups=mask_preds.shape[1])
#     grad_y = F.conv2d(mask_preds, sobel_y, padding=1, groups=mask_preds.shape[1])
    
#     grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
#     return grad_magnitude


# # def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
# #     """Estimate uncertainty based on pred logits."""
# #     if mask_preds.shape[1] == 1:
# #         gt_class_logits = mask_preds.clone()
# #     else:
# #         inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
# #         gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
# #     return torch.abs(gt_class_logits)  # Higher values mean higher uncertainty


# # From MMdet
# def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
#     """Estimate uncertainty based on pred logits.

#     We estimate uncertainty as L1 distance between 0.0 and the logits
#     prediction in 'mask_preds' for the foreground class in `classes`.

#     Args:
#         mask_preds (Tensor): mask predication logits, shape (num_rois,
#             num_classes, mask_height, mask_width).

#         labels (Tensor): Either predicted or ground truth label for
#             each predicted mask, of length num_rois.

#     Returns:
#         scores (Tensor): Uncertainty scores with the most uncertain
#             locations having the highest uncertainty score,
#             shape (num_rois, 1, mask_height, mask_width)
#     """
#     if mask_preds.shape[1] == 1:
#         gt_class_logits = mask_preds.clone()
#     else:
#         inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
#         gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
#     return -torch.abs(gt_class_logits) # more negative values --> more certain areas, more closer to zero --> more uncertain areas



# def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
#     """
#     Find `num_points` most uncertain points from `uncertainty_map` grid.

#     Args:
#         uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
#             values for a set of points on a regular H x W grid.
#         num_points (int): The number of points P to select.

#     Returns:
#         point_indices (Tensor): A tensor of shape (N, P) that contains indices from
#             [0, H x W) of the most uncertain points.
#         point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
#             coordinates of the most uncertain points from the H x W grid.
#     """
#     print('uncertainty_map.shape:', uncertainty_map.shape)
#     R, _, H, W = uncertainty_map.unsqueeze(1).shape
#     h_step = 1.0 / float(H)
#     w_step = 1.0 / float(W)

#     num_points = min(H * W, num_points)
#     point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
#     point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
#     point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
#     point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
#     return point_indices, point_coords


# def get_uncertain_point_coords_with_randomness(
#         mask_preds: Tensor,  labels: Tensor, mask_gt: Tensor,num_points: int, 
#         oversample_ratio: float, importance_sample_ratio: float) -> Tensor:

#     """Get ``num_points`` most uncertain points with random points during
#     train.

#     Sample points in [0, 1] x [0, 1] coordinate space based on their
#     uncertainty. The uncertainties are calculated for each point using
#     'get_uncertainty()' function that takes point's logit prediction as
#     input.

#     Args:
#         mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
#             mask_height, mask_width) for class-specific or class-agnostic
#             prediction.
#         labels (Tensor): The ground truth class for each instance.
#         num_points (int): The number of points to sample.
#         oversample_ratio (float): Oversampling parameter.
#         importance_sample_ratio (float): Ratio of points that are sampled
#             via importnace sampling.

#     Returns:
#         point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
#             that contains the coordinates sampled points.
#     """
#     assert oversample_ratio >= 1
#     assert 0 <= importance_sample_ratio <= 1
#     batch_size = mask_preds.shape[0]
#     num_sampled = int(num_points * oversample_ratio)
    
#     margin = 25
#     # point_coords = torch.rand(
#     #     batch_size, num_sampled, 2, device=mask_preds.device)
#     point_coords = process_masks_and_boxes(mask_preds, mask_gt, margin , num_sampled) 
    
#     # print('point_coords.shape:', point_coords.shape, 'mask_preds.shape:', mask_preds.shape)
#     # point_logits = point_sample(mask_preds, point_coords)
#     # It is crucial to calculate uncertainty based on the sampled
#     # prediction value for the points. Calculating uncertainties of the
#     # coarse predictions first and sampling them for points leads to
#     # incorrect results.  To illustrate this: assume uncertainty func(
#     # logits)=-abs(logits), a sampled point between two coarse
#     # predictions with -1 and 1 logits has 0 logits, and therefore 0
#     # uncertainty value. However, if we calculate uncertainties for the
#     # coarse predictions first, both will have -1 uncertainty,
#     # and sampled point will get -1 uncertainty.
#     # point_uncertainties = get_uncertainty(point_logits, labels)
#     # if mask_preds.numel() != 0:
#     #     create_uncertainty_heatmap(mask_preds, point_coords, point_uncertainties)
    
#     num_uncertain_points = int(importance_sample_ratio * num_points)
#     num_random_points = num_points - num_uncertain_points
    
#     # idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
#     # shift = num_sampled * torch.arange(
#     #     batch_size, dtype=torch.long, device=mask_preds.device)
#     # idx += shift[:, None] # uncertain_indices
#     # # point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
#     # #     batch_size, num_uncertain_points, 2)
    
#     # uncertain_point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
#     #     batch_size, num_uncertain_points, 2)
    
#     # Randomly selecting uncertain points instead of based on uncertainty
#     total_points = point_coords.shape[1]  # point_coords: [batch_size, num_sampled, 2]
#     random_indices = torch.randperm(total_points)[:num_uncertain_points]
    
#     # Extract uncertain points randomly
#     uncertain_point_coords = point_coords[:, random_indices, :]
    
#     if num_random_points > 0:
#         rand_roi_coords = torch.rand(
#             batch_size, num_random_points, 2, device=mask_preds.device)
#         # point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
#         point_coords = torch.cat((uncertain_point_coords, rand_roi_coords), dim=1)
        
#     return point_coords, uncertain_point_coords, rand_roi_coords


# # From MMcv
# def normalize(grid):
#     """Normalize input grid from [-1, 1] to [0, 1]
#     Args:
#         grid (torch.Tensor): The grid to be normalized, range [-1, 1].
#     Returns:
#         torch.Tensor: Normalized grid, range [0, 1].
#     """
#     return (grid + 1.0) / 2.0

# def denormalize(grid):
#     """Denormalize input grid from range [0, 1] to [-1, 1]
#     Args:
#         grid (torch.Tensor): The grid to be denormalized, range [0, 1].
#     Returns:
#         torch.Tensor: Denormalized grid, range [-1, 1].
#     """
#     return grid * 2.0 - 1.0

# def point_sample(input: Tensor,
#                  points: Tensor,
#                  align_corners: bool = False,
#                  **kwargs) -> Tensor:
#     """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
#     Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
#     lie inside ``[0, 1] x [0, 1]`` square.

#     Args:
#         input (torch.Tensor): Feature map, shape (N, C, H, W).
#         points (torch.Tensor): Image based absolute point coordinates
#             (normalized), range [0, 1] x [0, 1], shape (N, P, 2) or
#             (N, Hgrid, Wgrid, 2).
#         align_corners (bool, optional): Whether align_corners.
#             Default: False

#     Returns:
#         torch.Tensor: Features of `point` on `input`, shape (N, C, P) or
#         (N, C, Hgrid, Wgrid).
#     """

#     add_dim = False
#     if points.dim() == 3:
#         add_dim = True
#         points = points.unsqueeze(2)
    
#     points = points.to(input.device)
#     output = F.grid_sample(
#         input, denormalize(points), align_corners=align_corners, **kwargs)
#     if add_dim:
#         output = output.squeeze(3)
#     return output

# def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
#     """
#     Find `num_points` most uncertain points from `uncertainty_map` grid.

#     Args:
#         uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
#             values for a set of points on a regular H x W grid.
#         num_points (int): The number of points P to select.

#     Returns:
#         point_indices (Tensor): A tensor of shape (N, P) that contains indices from
#             [0, H x W) of the most uncertain points.
#         point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
#             coordinates of the most uncertain points from the H x W grid.
#     """
#     R, _, H, W = uncertainty_map.unsqueeze(1).shape
#     h_step = 1.0 / float(H)
#     w_step = 1.0 / float(W)

#     num_points = min(H * W, num_points)
#     point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
#     point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
#     point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
#     point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
#     return point_indices, point_coords


from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
import os

def draw_and_save_images(masks, boxes, expanded_boxes, output_dir = '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_debugging/output/box_vis'):
    # print('masks.shape:', masks.shape, 'boxes:', boxes.shape, 'expanded_boxes:', expanded_boxes.shape)
    for i, (mask, box, exp_box) in enumerate(zip(masks, boxes, expanded_boxes)):
        # Ensure tensors are on CPU and convert to numpy for plotting
        mask_array = mask.cpu().numpy().squeeze()
        box = box.cpu().numpy()
        exp_box = exp_box.cpu().numpy().squeeze()
        # min_box = min_box.cpu().numpy().squeeze()
        # print('mask_array.shape:', mask_array.shape, 'box:', box.shape, 'exp_box:', exp_box.shape)
        

        # Setup plot
        fig, ax = plt.subplots()
        ax.imshow(mask_array, cmap='gray')
        
        # Draw original bounding box - Red
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        # Draw expanded bounding box - Blue, with dashed style
        rect_exp = plt.Rectangle((exp_box[0], exp_box[1]), exp_box[2] - exp_box[0], exp_box[3] - exp_box[1], linewidth=2, edgecolor='b', linestyle='--', facecolor='none')
        ax.add_patch(rect_exp)
        
        # Draw expanded bounding box - Blue, with dashed style
        # rect_min = plt.Rectangle((min_box[0],min_box[1]), min_box[2] - min_box[0], min_box[3] - min_box[1], linewidth=2, edgecolor='r', linestyle='--', facecolor='none')
        # ax.add_patch(rect_min)
        
        # Set title and remove axes
        ax.set_title('Mask with Bounding Boxes')
        ax.axis('off')

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{output_dir}/mask_with_boxes_{i}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()


def calculate_bbox_from_mask(mask):
    
    pos = torch.where(mask)
    if pos[0].numel() == 0 or pos[1].numel() == 0:
        # Return an empty tensor with the correct shape and device
        return torch.tensor([], dtype=torch.float32, device=mask.device)
    
    # Use torch.any() to find rows and columns that contain at least one True value
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    if not rows.any() or not cols.any():
        return None  # Handle cases where there are no objects

    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    # return torch.tensor([xmin*mask.shape[0], ymin*mask.shape[1], xmax*mask.shape[0], ymax*mask.shape[1]], device=mask.device)
    return torch.tensor([xmin, ymin, xmax, ymax], device=mask.device)


# def add_margin_to_boxes(boxes, margin, image_width, image_height):
#     """Add margins to boxes to create an expanded and a contracted box, ensuring they don't go beyond image boundaries."""
#     margin_max, margin_min = margin
#     # Clone original boxes to create new sets for modification
#     new_boxes_max = boxes.clone()
#     new_boxes_min = boxes.clone()

#     # Applying maximum margin to expand the boxes
#     new_boxes_max[:, 0] = torch.clamp(new_boxes_max[:, 0] - margin_max, min=0)  # x_min
#     new_boxes_max[:, 1] = torch.clamp(new_boxes_max[:, 1] - margin_max, min=0)  # y_min
#     new_boxes_max[:, 2] = torch.clamp(new_boxes_max[:, 2] + margin_max, max=image_width - 1)  # x_max
#     new_boxes_max[:, 3] = torch.clamp(new_boxes_max[:, 3] + margin_max, max=image_height - 1)  # y_max

#     # Applying minimum margin to contract the boxes
#     new_boxes_min[:, 0] = torch.clamp(new_boxes_min[:, 0] + margin_min, min=0)  # x_min
#     new_boxes_min[:, 1] = torch.clamp(new_boxes_min[:, 1] + margin_min, min=0)  # y_min
#     new_boxes_min[:, 2] = torch.clamp(new_boxes_min[:, 2] - margin_min, max=image_width - 1)  # x_max
#     new_boxes_min[:, 3] = torch.clamp(new_boxes_min[:, 3] - margin_min, max=image_height - 1)  # y_max

#     # Ensure the minimum values are still less than the maximum values
#     new_boxes_max[:, 2] = torch.max(new_boxes_max[:, 2], new_boxes_max[:, 0] + 1)
#     new_boxes_max[:, 3] = torch.max(new_boxes_max[:, 3], new_boxes_max[:, 1] + 1)
#     new_boxes_min[:, 2] = torch.max(new_boxes_min[:, 2], new_boxes_min[:, 0] + 1)
#     new_boxes_min[:, 3] = torch.max(new_boxes_min[:, 3], new_boxes_min[:, 1] + 1)

#     return new_boxes_max, new_boxes_min

def add_margin_to_boxes(boxes, margin, image_width, image_height):
    """Add margins to boxes to create an expanded and a contracted box, ensuring they don't go beyond image boundaries."""
    margin_max = margin
    # Clone original boxes to create new sets for modification
    new_boxes_max = boxes.clone()

    # Applying maximum margin to expand the boxes
    new_boxes_max[:, 0] = torch.clamp(new_boxes_max[:, 0] - margin_max, min=0)  # x_min
    new_boxes_max[:, 1] = torch.clamp(new_boxes_max[:, 1] - margin_max, min=0)  # y_min
    new_boxes_max[:, 2] = torch.clamp(new_boxes_max[:, 2] + margin_max, max=image_width - 1)  # x_max
    new_boxes_max[:, 3] = torch.clamp(new_boxes_max[:, 3] + margin_max, max=image_height - 1)  # y_max


    # Ensure the minimum values are still less than the maximum values
    new_boxes_max[:, 2] = torch.max(new_boxes_max[:, 2], new_boxes_max[:, 0] + 1)
    new_boxes_max[:, 3] = torch.max(new_boxes_max[:, 3], new_boxes_max[:, 1] + 1)

    return new_boxes_max

def sample_points_from_boxes(boxes, num_points, image_width, image_height):
    """Sample points from expanded boxes ensuring valid ranges."""
    batch_size = boxes.shape[0]
    points = []

    for i in range(batch_size):
        x_min, y_min, x_max, y_max = boxes[i, :]

        # Ensuring min and max are correctly ordered and valid
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        # Sampling points within the box
        x_coords = torch.randint(low=int(x_min), high=int(x_max), size=(num_points,))
        y_coords = torch.randint(low=int(y_min), high=int(y_max), size=(num_points,))

        points.append(torch.stack((x_coords.float() / image_width, y_coords.float() / image_height), dim=1))
    
    return torch.stack(points, dim=0)

def process_masks_and_boxes(mask_preds, mask_gt, margin, num_points):
    mask_gt_orig= mask_gt
    batch_size, _, height, width = mask_preds.shape
    sampled_points = []
    mask = mask_gt.squeeze(1)
    # print('gt_unique:', torch.unique(mask))
    
    # print('mask_gt.shape:', mask_gt.shape, 'mask.shape:', mask.shape)
    # all_boxes = []
    # all_expanded_boxes = []
    # all_minimized_boxes = []
    for i in range(batch_size):
        mask_gt = mask[i]  # Assuming mask_preds is [batch_size, channels, height, width]
        # print('mask_gt.shape after:', mask_gt.shape)
        box_pred = calculate_bbox_from_mask(mask_preds.squeeze(1)[i] > 0.55)
        
        if box_pred.nelement() == 0:
            # No valid box, sample points randomly across the entire mask
            points = torch.rand(1,num_points, 2) * torch.tensor([width, height])
        else:
            # Add margin to the box
            expanded_box = add_margin_to_boxes(box_pred.unsqueeze(0),margin, width, height)

            # Sample points from the expanded box
            points = sample_points_from_boxes(expanded_box, num_points, width, height)
            # print('points.shape:', points.shape)
            
        sampled_points.append(points)
        # all_boxes.append(box_pred)
        # all_expanded_boxes.append(expanded_box)
        # all_minimized_boxes.append(minimized_box)
    # draw_and_save_images(mask_preds, all_boxes, all_expanded_boxes)
    if sampled_points:
        sampled_points = torch.cat(sampled_points, dim=0)
    else:
        sampled_points = torch.empty((0, num_points, 2), device=mask_preds.device)

    return sampled_points.to(mask_preds.device)



def calculate_uncertainty(sem_seg_logits):
    # Check if there are at least 2 elements along dimension 1
    if sem_seg_logits.size(1) < 2:
        # Handle special case where top-2 cannot be applied
        # For instance, return the absolute value if only 1 class exists
        top2_scores = torch.abs(sem_seg_logits)  # Or any suitable operation
    else:
        # Calculate top-2 scores as intended
        top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    
    # Assuming uncertainty is based on the difference in scores
    uncertainty = top2_scores[:, 0] - top2_scores[:, 1] if sem_seg_logits.size(1) >= 2 else top2_scores.squeeze(1)
    return uncertainty


def calculate_gradients(mask_preds):
    # Assuming mask_preds is a tensor of shape (N, C, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask_preds.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask_preds.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(mask_preds, sobel_x, padding=1, groups=mask_preds.shape[1])
    grad_y = F.conv2d(mask_preds, sobel_y, padding=1, groups=mask_preds.shape[1])
    
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude


# def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
#     """Estimate uncertainty based on pred logits."""
#     if mask_preds.shape[1] == 1:
#         gt_class_logits = mask_preds.clone()
#     else:
#         inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
#         gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
#     return torch.abs(gt_class_logits)  # Higher values mean higher uncertainty


# From MMdet
# def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
#     """Estimate uncertainty based on pred logits.

#     We estimate uncertainty as L1 distance between 0.0 and the logits
#     prediction in 'mask_preds' for the foreground class in `classes`.

#     Args:
#         mask_preds (Tensor): mask predication logits, shape (num_rois,
#             num_classes, mask_height, mask_width).

#         labels (Tensor): Either predicted or ground truth label for
#             each predicted mask, of length num_rois.

#     Returns:
#         scores (Tensor): Uncertainty scores with the most uncertain
#             locations having the highest uncertainty score,
#             shape (num_rois, 1, mask_height, mask_width)
#     """
#     if mask_preds.shape[1] == 1:
#         gt_class_logits = mask_preds.clone()
#     else:
#         inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
#         gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
#     return -torch.abs(gt_class_logits) # more negative values --> more certain areas, more closer to zero --> more uncertain areas

def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
    """Estimate uncertainty based on probabilistic mask predictions.

    We estimate uncertainty as the L1 distance from 0.5 for probabilistic
    predictions in 'mask_preds' for the foreground class in `labels`.

    Args:
        mask_preds (Tensor): Mask prediction probabilities, shape (num_rois,
            num_classes, mask_height, mask_width). Expected to be in range [0, 1].

        labels (Tensor): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score (values close to 0.5),
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_preds.shape[1] == 1:
        # If there's only one class, directly use mask_preds as probabilistic masks
        gt_class_probs = mask_preds.clone()
    else:
        # Select the probability map corresponding to the given label for each mask
        inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
        gt_class_probs = mask_preds[inds, labels].unsqueeze(1)

    # Calculate uncertainty as the distance from 0.5
    uncertainty_scores = -torch.abs(gt_class_probs - 0.5)
    
    return uncertainty_scores


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    print('uncertainty_map.shape:', uncertainty_map.shape)
    R, _, H, W = uncertainty_map.unsqueeze(1).shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


def get_uncertain_point_coords_with_randomness(
        mask_preds: Tensor,  labels: Tensor, mask_gt: Tensor,num_points: int, 
        oversample_ratio: float, importance_sample_ratio: float) -> Tensor:
    
    def create_uncertainty_heatmap(mask_preds, point_coords, point_uncertainties, index=0, output_dir='/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_debugging/output/uncertainty_heatmap'):
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Extract the mask and uncertainties for the specified index
        mask = mask_preds[index].squeeze().cpu().numpy()
        points = point_coords[index].cpu().numpy()
        uncertainties = point_uncertainties[index].squeeze().cpu().numpy()
        # print('mask.shape:', mask.shape, 'points.shape:', points.shape, 'uncertainties.shape:', uncertainties.shape)
        # Convert points to image coordinates
        points[:, 0] *= mask.shape[0]  # Scale y coordinates
        points[:, 1] *= mask.shape[1]  # Scale x coordinates

        # Create a grid for interpolation
        grid_x, grid_y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))

        # Interpolate the uncertainties to create a full grid
        grid_z = griddata(points, uncertainties, (grid_x, grid_y), method='cubic')

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original Mask')
        ax[0].axis('off')

        # Display the uncertainty heatmap
        heatmap = ax[1].imshow(grid_z, cmap='viridis',  aspect='equal')
        ax[1].set_title('Uncertainty Heatmap')
        ax[1].axis('off')

        # Adding a color bar to the heatmap
        cbar = plt.colorbar(heatmap, ax=ax[1], orientation='vertical')
        cbar.set_label('Level of Uncertainty', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=10)  # Adjust to suit your aesthetic needs

        # Save the figure
        filename = f"Uncertainty_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close(fig)

    
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (Tensor): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_preds.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    
    # margin = 40
    margin = 15
    # point_coords = torch.rand(
    #     batch_size, num_sampled, 2, device=mask_preds.device)
    point_coords = process_masks_and_boxes(mask_preds, mask_gt, margin , num_sampled) 
    
    # print('point_coords.shape:', point_coords.shape, 'mask_preds.shape:', mask_preds.shape)
    point_logits = point_sample(mask_preds, point_coords)
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = get_uncertainty(point_logits, labels)
    # if mask_preds.numel() != 0:
    #     create_uncertainty_heatmap(mask_preds, point_coords, point_uncertainties)
    
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_preds.device)
    idx += shift[:, None] # uncertain_indices
    # point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
    #     batch_size, num_uncertain_points, 2)
    
    uncertain_point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2)
    
    
    if num_random_points > 0:
        rand_roi_coords = torch.rand(
            batch_size, num_random_points, 2, device=mask_preds.device)
        # point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
        point_coords = torch.cat((uncertain_point_coords, rand_roi_coords), dim=1)
        
    return point_coords, uncertain_point_coords, rand_roi_coords


# From MMcv
def normalize(grid):
    """Normalize input grid from [-1, 1] to [0, 1]
    Args:
        grid (torch.Tensor): The grid to be normalized, range [-1, 1].
    Returns:
        torch.Tensor: Normalized grid, range [0, 1].
    """
    return (grid + 1.0) / 2.0

def denormalize(grid):
    """Denormalize input grid from range [0, 1] to [-1, 1]
    Args:
        grid (torch.Tensor): The grid to be denormalized, range [0, 1].
    Returns:
        torch.Tensor: Denormalized grid, range [-1, 1].
    """
    return grid * 2.0 - 1.0

def point_sample(input: Tensor,
                 points: Tensor,
                 align_corners: bool = False,
                 **kwargs) -> Tensor:
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (torch.Tensor): Feature map, shape (N, C, H, W).
        points (torch.Tensor): Image based absolute point coordinates
            (normalized), range [0, 1] x [0, 1], shape (N, P, 2) or
            (N, Hgrid, Wgrid, 2).
        align_corners (bool, optional): Whether align_corners.
            Default: False

    Returns:
        torch.Tensor: Features of `point` on `input`, shape (N, C, P) or
        (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    
    points = points.to(input.device)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.unsqueeze(1).shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords






