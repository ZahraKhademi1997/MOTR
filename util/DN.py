import torch
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from models.structures import Instances


# def prepare_targets(targets, images_height, images_width):
#         h_pad, w_pad = images_height, images_width
#         new_targets = []
#         for targets_per_image in targets:
#             # pad gt
#             h, w = targets_per_image.image_size
#             image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

#             gt_masks = targets_per_image.gt_masks
#             padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
#             padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
#             new_targets.append(
#                 {
#                     "labels": targets_per_image.gt_classes,
#                     "masks": padded_masks,
#                     "boxes":box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
#                 }
#             )
#         return new_targets
def prepare_targets(targets, images_height, images_width):
        h_pad, w_pad = images_height, images_width
        new_targets = []
        targets_per_image = targets
        h, w = targets_per_image.image_size
        image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)

        gt_masks = targets_per_image.get('masks')
        padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
        new_targets.append(
            {
                "labels": targets_per_image.get('labels'),
                "masks": padded_masks,
                "boxes":box_xyxy_to_cxcywh(targets_per_image.get('boxes'))/(image_size_xyxy.to(gt_masks.device))
            }
        )
        return new_targets


# def empty_GT_instances(gt_instances, n_repeats=6):
#     gt_instances_new = Instances((1, 1))
#     # Calculate the length of the new Instances
#     length = len(gt_instances) * n_repeats
#     print('Length of new gt_instances:', length)
    
#     # Creating empty tensors to populate new Instances
#     obj_ids = torch.full((length,), -1, dtype=torch.long, device= gt_instances.get('masks').device)
#     area = torch.zeros(length, dtype=torch.float, device=gt_instances.get('masks').device)
#     boxes = torch.zeros((length, 4), dtype=torch.float, device= gt_instances.get('masks').device)
#     labels = torch.zeros(length, dtype=torch.int64, device=gt_instances.get('masks').device)  
#     masks = torch.zeros((length,  gt_instances.get('masks').shape[1],  gt_instances.get('masks').shape[2]), dtype=torch.uint8, device= gt_instances.get('masks').device)  

#     # Assuming 'Instances' can be initialized with a list of fields
#     image_height=gt_instances.get('masks').shape[1]
#     image_width=gt_instances.get('masks').shape[2]
#     image_size = (image_height, image_width)
#     # gt_instances_new.num_instances=length
#     gt_instances_new.image_height=image_height
#     gt_instances_new.image_width=image_width
#     # gt_instances_new.image_size=image_size
#     gt_instances_new.fields=[
#         {'name': 'obj_ids', 'data': obj_ids},
#         {'name': 'area', 'data': area},
#         {'name': 'boxes', 'data': boxes},
#         {'name': 'labels', 'data': labels},
#         {'name': 'masks', 'data': masks}
#     ]

#     return gt_instances_new

# @staticmethod
def _targets_to_instances(targets: dict, img_shape) -> Instances:
    gt_instances = Instances(tuple(img_shape))
    gt_instances.masks = targets['masks']
    gt_instances.boxes = targets['boxes']
    gt_instances.labels = targets['labels']
    gt_instances.obj_ids = targets['obj_ids']
    gt_instances.area = targets['area']
    return gt_instances
       
def repeat_GT_instances(gt_instances, n_repeats=6):
    # Accessing and repeating each attribute
    targets = {}
    repeated_masks = gt_instances.get('masks').repeat_interleave(n_repeats, dim=0)
    repeated_boxes = gt_instances.get('boxes').repeat(n_repeats, 1)
    repeated_labels = gt_instances.get('labels').repeat(n_repeats)
    repeated_obj_ids = gt_instances.get('obj_ids').repeat(n_repeats)
    repeated_area = gt_instances.get('area').repeat(n_repeats)
    
    targets['masks'] = repeated_masks
    targets['boxes'] = repeated_boxes
    targets['labels'] = repeated_labels
    targets['obj_ids'] = repeated_obj_ids
    targets['area'] = repeated_area
    
    gt_instances_new = _targets_to_instances(targets, (gt_instances.get('masks').shape[1], gt_instances.get('masks').shape[2]))
    # print('gt_instances_new:', gt_instances_new)
    return gt_instances_new