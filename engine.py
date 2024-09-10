# # ------------------------------------------------------------------------
# # Copyright (c) 2021 megvii-model. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # ------------------------------------------------------------------------


# """
# Train and eval functions used in main.py
# """
# import cv2
# import math
# import numpy as np
# import os
# import sys
# from typing import Iterable

# import torch
# import util.misc as utils
# from util import box_ops
# from collections import OrderedDict
# from torch import Tensor
# from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator
# from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda
# from torch.utils.tensorboard import SummaryWriter
# from torch.autograd import profiler
# import matplotlib.pyplot as plt


# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
#     model.train()
#     criterion.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10

#     prefetcher = data_prefetcher(data_loader, device, prefetch=True)
#     samples, targets = prefetcher.next()

#     # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#     for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
#         outputs = model(samples)

#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict
#         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

#         loss_value = losses_reduced_scaled.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         losses.backward()
#         if max_norm > 0:
#             grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#         else:
#             grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
#         optimizer.step()

#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(grad_norm=grad_total_norm)

#         samples, targets = prefetcher.next()
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# #########################################################################################################################
# # () Training whole part of the model
# def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
#     # print("Entered train_one_epoch_mot in engine")
#     model.train()
#     # print('model.train in engine')
#     criterion.train()
#     # print('criterion.train in engine')
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10

#     # Dictionary to accumulate loss values for each key over iterations
#     loss_accumulator = {key: 0.0 for key in criterion.weight_dict.keys()}
    
#     #####################################################################
#     # () Defining function to log gradient of different part of the model 
#     writer_grad = SummaryWriter('/home/zahra/Documents/Projects/prototype/MOTR-codes/test_mask/MOTR-MOTR_version2_mask_applemots/output/logs_grad') 
#     def log_gradients(model, writer_grad, epoch, iteration):
#         # Check if the model is wrapped in DistributedDataParallel
#         if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#             model = model.module

#         modules = {
#             'backbone': model.backbone,
#             'encoder': model.transformer.encoder,
#             'decoder': model.transformer.decoder,
#             'input_proj': model.input_proj,
#             'PerPixelEmbedding' : model.PerPixelEmbedding,
#             'seg_branches' : model.seg_branches,
#             'AxialBlock' : model.AxialBlock,
#             # 'mask_positional_encoding' : model.mask_positional_encoding,
#             # 'spatial_mlp' : model.spatial_mlp,
#             'pos_cross_attention' : model.pos_cross_attention,
#             'post_process': model.post_process,
            
#         }

#         for name, module in modules.items():
#             total_grad_norm = 0
#             for param in module.parameters():
#                 if param.grad is not None:
#                     total_grad_norm += param.grad.data.norm(2).item()
#             writer_grad.add_scalar(f'grad_norm/{name}', total_grad_norm, epoch * len(data_loader) + iteration)

#     iteration = 0
#     #####################################################################
#     # def forward_hook(module, input, output):
#     #     # Log the output of each layer (or selected layers) during the forward pass
#     #     writer.add_histogram(f'{module.__class__.__name__}_output', output, epoch)
    
#     # last_layer = None

#     # # Iterate through all modules to find the last one
#     # for module in model.modules():
#     #     last_layer = module

#     # # Check if the last layer is found
#     # if last_layer is not None:
#     #     # Attach the forward hook to the last layer
#     #     last_layer.register_forward_hook(forward_hook)
#     #     print(f"Hook attached to the last layer: {last_layer.__class__.__name__}")
#     # else:
#     #     print("No layer found in the model")
        
        
            
#     # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#     for data_dict in metric_logger.log_every(data_loader, print_freq, header):
#         data_dict = data_dict_to_cuda(data_dict, device)
#         outputs = model(data_dict)
        
        
#         loss_dict = criterion(outputs, data_dict)
#         weight_dict = criterion.weight_dict
#         for key, value in loss_dict.items():
#             value.requires_grad_(True)
            
#         # print('data_dict is:', data_dict)
#         # print('outputs are:', outputs)
#         # print('loss_dict is:', loss_dict)
#         # print('weight_dict is:', weight_dict)
#         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#         losses.requires_grad_(True) 
        
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#         #                               for k, v in loss_dict_reduced.items()}
        
        
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
        
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
#         # print('#$#$#$#$#$#$#$#$#$#$losses reduced value in engine.py is:', losses_reduced_scaled)
#         loss_value = losses_reduced_scaled.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)

#         #####################################################################################
#         # Store initial weights and biases
#         # initial_k_linear_weights = model.module.bbox_attention.k_linear.weight.clone().detach()
#         # initial_k_linear_bias = model.module.bbox_attention.k_linear.bias.clone().detach()
#         # initial_q_linear_weights = model.module.bbox_attention.q_linear.weight.clone().detach()
#         # initial_q_linear_bias = model.module.bbox_attention.q_linear.bias.clone().detach()
#         #####################################################################################
        
#         optimizer.zero_grad()
        
#         # grads = {}
#         # def save_grad(name):
#         #     def hook(grad):
#         #         grads[name] = grad
#         #     return hook

#         # for name, param in model.named_parameters():
#         #     if param.requires_grad:
#         #         param.register_hook(save_grad(name)) 
        
#         losses.backward()
#         # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
#         #     losses.backward()
#         # profiling_results = prof.key_averages().table(sort_by="self_cpu_time_total")
#         # output_dir="/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/outputs/gradients.txt"
#         # with open (output_dir, 'w') as f:
#         #     f.write(profiling_results) 
            
        
#         ########################################################################
#         # () Logging gradients
#         log_gradients(model, writer_grad, epoch, iteration)
#         ########################################################################
        
#         ################################################### 
#         # ()
#         # for name, param in model.named_parameters():
#         #     if param.requires_grad and param.grad is not None:
#         #         print(f"Gradient for {name}: {param.grad.norm().item()}")  # Print the norm of the gradients
#         #     else:
#         #         print(f"No gradient or not trainable for {name}")
        
#         # for name, grad in grads.items():
#         #     if grad is not None:
#         #         plt.hist(grad.cpu().numpy().flatten(), bins=100)
#         #         plt.title(f'Gradient histogram for {name}')
#         #         plt.xlabel('Gradient values')
#         #         plt.ylabel('Frequency')
#         #         plt.savefig(f'/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/outputs/gradients/gradient_histogram_{name}.png')
#         #         plt.close()
#         ################################################### 
        
#         if max_norm > 0:
#             grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#         else:
#             grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
#         optimizer.step()
        
#         ####################################################################################################################
#         # Check if weights and biases have changed
#         # k_linear_weights_changed = not torch.equal(initial_k_linear_weights, model.module.bbox_attention.k_linear.weight)
#         # k_linear_bias_changed = not torch.equal(initial_k_linear_bias, model.module.bbox_attention.k_linear.bias)
#         # q_linear_weights_changed = not torch.equal(initial_q_linear_weights, model.module.bbox_attention.q_linear.weight)
#         # q_linear_bias_changed = not torch.equal(initial_q_linear_bias, model.module.bbox_attention.q_linear.bias)

#         # print(f"k_linear weights changed: {k_linear_weights_changed}")
#         # print(f"k_linear bias changed: {k_linear_bias_changed}")
#         # print(f"q_linear weights changed: {q_linear_weights_changed}")
#         # print(f"q_linear bias changed: {q_linear_bias_changed}")
#         ####################################################################################################################

#         for key, value in loss_dict.items():
#             loss_accumulator[key] += value.item()

#         # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
#         # metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(grad_norm=grad_total_norm)
#         # gather the stats from all processes
#         iteration+=1
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     writer_grad.close()
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# #########################################################################################################################


# #########################################################################################################################
# # () Freezing all layers except for the segmentation head
# # def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
# #                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
# #                     device: torch.device, epoch: int, max_norm: float = 0):
    
# #     #####################################################################
# #     # () Freezing an loading checkpoint    
# #     # Freeze all layers first
# #     for param in model.parameters():
# #         param.requires_grad = False

# #     # # Load checkpoint if provided
# #     # checkpoint_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/output/checkpoint0025.pth"
# #     # checkpoint = torch.load(checkpoint_path)
    
# #     # # Adjust the keys in the state dictionary
# #     # new_state_dict = OrderedDict()
# #     # for k, v in checkpoint['model'].items():
# #     #     # Prepend 'module.' to the keys
# #     #     name = 'module.' + k
# #     #     new_state_dict[name] = v

# #     # # Load the adjusted state dict
# #     # model.load_state_dict(new_state_dict)

# #     # output_dir = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/outputs/checkpoint.txt"
# #     # with open (output_dir, "w") as f:
# #     #     f.write(str(checkpoint))
# #     # print("checkpoint.keys are:", checkpoint.keys()) 
# #     # model.load_state_dict(checkpoint['model'])

# #     # Unfreeze bbox_attention and mask_head
# #     for param in model.module.bbox_attention.parameters():
# #         param.requires_grad = True
# #     for param in model.module.mask_head.parameters():
# #         param.requires_grad = True
# #     #####################################################################  
        
# #     model.train()
# #     criterion.train()
# #     metric_logger = utils.MetricLogger(delimiter="  ")
# #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# #     # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
# #     metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
# #     header = 'Epoch: [{}]'.format(epoch)
# #     print_freq = 10

# #     # Dictionary to accumulate loss values for each key over iterations
# #     loss_accumulator = {key: 0.0 for key in criterion.weight_dict.keys()}
    
# #     #####################################################################
# #     # () Defining function to log gradient of different part of the model 
# #     writer = SummaryWriter('/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/outputs/logs_grad') 
# #     def log_gradients(model, writer, epoch, iteration):
# #         # Check if the model is wrapped in DistributedDataParallel
# #         if isinstance(model, torch.nn.parallel.DistributedDataParallel):
# #             model = model.module

# #         modules = {
# #             'backbone': model.backbone,
# #             'position_embedding': model.backbone[1],
# #             'encoder': model.transformer.encoder,
# #             'decoder': model.transformer.decoder,
# #             'input_proj': model.input_proj,
# #             'bbox_attention/q_linear': model.bbox_attention.q_linear,
# #             'bbox_attention/k_linear': model.bbox_attention.k_linear,
# #             'mask_head/lay1': model.mask_head.lay1,
# #             'mask_head/gn1': model.mask_head.gn1,
# #             'mask_head/lay2': model.mask_head.lay2,
# #             'mask_head/gn2': model.mask_head.gn2,
# #             'mask_head/lay3': model.mask_head.lay3,
# #             'mask_head/gn3': model.mask_head.gn3,
# #             'mask_head/lay4': model.mask_head.lay4,
# #             'mask_head/gn4': model.mask_head.gn4,
# #             'mask_head/lay5': model.mask_head.lay5,
# #             'mask_head/gn5': model.mask_head.gn5,
# #             'mask_head/out_lay': model.mask_head.out_lay,
# #             'mask_head/adapter1': model.mask_head.adapter1,
# #             'mask_head/adapter2': model.mask_head.adapter2,
# #             'mask_head/adapter3': model.mask_head.adapter3,
# #             'postprocessor': model.postprocessor,
# #             'post_process': model.post_process,
# #             'criterion': model.criterion
# #         }

# #         for name, module in modules.items():
# #             total_grad_norm = 0
# #             for param in module.parameters():
# #                 if param.grad is not None:
# #                     total_grad_norm += param.grad.data.norm(2).item()
# #             writer.add_scalar(f'grad_norm/{name}', total_grad_norm, epoch * len(data_loader) + iteration)

# #     iteration = 0
# #     #####################################################################
              
# #     # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
# #     for data_dict in metric_logger.log_every(data_loader, print_freq, header):
# #         data_dict = data_dict_to_cuda(data_dict, device)
# #         outputs = model(data_dict)
# #         loss_dict = criterion(outputs, data_dict)
# #         weight_dict = criterion.weight_dict
# #         for key, value in loss_dict.items():
# #             value.requires_grad_(True)
# #         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
# #         losses.requires_grad_(True) 
        
# #         # reduce losses over all GPUs for logging purposes
# #         loss_dict_reduced = utils.reduce_dict(loss_dict)
# #         # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
# #         #                               for k, v in loss_dict_reduced.items()}
        
        
# #         loss_dict_reduced_scaled = {k: v * weight_dict[k]
# #                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
        
# #         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
# #         loss_value = losses_reduced_scaled.item()

# #         if not math.isfinite(loss_value):
# #             print("Loss is {}, stopping training".format(loss_value))
# #             print(loss_dict_reduced)
# #             sys.exit(1)
        

# #         #####################################################################################
# #         # Store initial weights and biases
# #         initial_k_linear_weights = model.module.bbox_attention.k_linear.weight.clone().detach()
# #         initial_k_linear_bias = model.module.bbox_attention.k_linear.bias.clone().detach()
# #         initial_q_linear_weights = model.module.bbox_attention.q_linear.weight.clone().detach()
# #         initial_q_linear_bias = model.module.bbox_attention.q_linear.bias.clone().detach()
# #         #####################################################################################
            
# #         optimizer.zero_grad()
# #         losses.backward()
        
# #         ########################################################################
# #         # () Logging gradients
# #         log_gradients(model, writer, epoch, iteration)
# #         ########################################################################
        
# #         ################################################### 
# #         # ()
# #         # for name, param in model.named_parameters():
# #         #     if param.requires_grad and param.grad is not None:
# #         #         print(f"Gradient for {name}: {param.grad.norm().item()}")  # Print the norm of the gradients
# #         #     else:
# #         #         print(f"No gradient or not trainable for {name}")
# #         ################################################### 
        
# #         if max_norm > 0:
# #             grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
# #         else:
# #             grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
# #         optimizer.step()
        
# #         ####################################################################################################################
# #         # Check if weights and biases have changed
# #         k_linear_weights_changed = not torch.equal(initial_k_linear_weights, model.module.bbox_attention.k_linear.weight)
# #         k_linear_bias_changed = not torch.equal(initial_k_linear_bias, model.module.bbox_attention.k_linear.bias)
# #         q_linear_weights_changed = not torch.equal(initial_q_linear_weights, model.module.bbox_attention.q_linear.weight)
# #         q_linear_bias_changed = not torch.equal(initial_q_linear_bias, model.module.bbox_attention.q_linear.bias)

# #         print(f"k_linear weights changed: {k_linear_weights_changed}")
# #         print(f"k_linear bias changed: {k_linear_bias_changed}")
# #         print(f"q_linear weights changed: {q_linear_weights_changed}")
# #         print(f"q_linear bias changed: {q_linear_bias_changed}")
# #         ####################################################################################################################


# #         for key, value in loss_dict.items():
# #             loss_accumulator[key] += value.item()

# #         # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
# #         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
# #         # metric_logger.update(class_error=loss_dict_reduced['class_error'])
# #         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
# #         metric_logger.update(grad_norm=grad_total_norm)
# #         # gather the stats from all processes
# #         iteration+=1
# #     metric_logger.synchronize_between_processes()
# #     print("Averaged stats:", metric_logger)
# #     writer.close()
# #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# #########################################################################################################################


# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
#     model.eval()
#     criterion.eval()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'

#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     coco_evaluator = CocoEvaluator(base_ds, iou_types)
#     # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )

#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         outputs = model(samples)
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#         #                               for k, v in loss_dict_reduced.items()}
#         # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#         #                      **loss_dict_reduced_scaled,
#         #                      **loss_dict_reduced_unscaled)
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              )
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes)
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name

#             panoptic_evaluator.update(res_pano)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_res = panoptic_evaluator.summarize()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#     return stats, coco_evaluator


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
Train and eval functions used in main.py
"""
import cv2
import math
import numpy as np
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from util import box_ops
from collections import OrderedDict
from torch import Tensor
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import profiler
import matplotlib.pyplot as plt


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



#########################################################################################################################
# () Training whole part of the model
def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # print("Entered train_one_epoch_mot in engine")
    model.train()
    # print('model.train in engine')
    criterion.train()
    # print('criterion.train in engine')
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Dictionary to accumulate loss values for each key over iterations
    loss_accumulator = {key: 0.0 for key in criterion.weight_dict.keys()}
    
    #####################################################################
    # () Defining function to log gradient of different part of the model 
    writer_grad = SummaryWriter('/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/logs/logs_grad') 
    def log_gradients(model, writer_grad, epoch, iteration):
        # Check if the model is wrapped in DistributedDataParallel
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        modules = {
            'backbone': model.backbone,
            'encoder': model.transformer.encoder,
            'decoder': model.transformer.decoder,
            'input_proj': model.input_proj,
            'PerPixelEmbedding' : model.PerPixelEmbedding,
            'track_embed' : model.track_embed,
            'transformer box_embed' : model.transformer.bbox_embed,
            'class_embed' : model.transformer.class_embed,
            'mask_embed' : model.transformer.mask_embed,
            'label_enc' : model.transformer.label_enc,
            'box_embed' : model.bbox_embed,
            'AxialBlock' : model.AxialBlock,
            'pos_cross_attention' : model.transformer.pos_cross_attention,
            # 'autoencoder' : model.transformer.autoencoder,
            # 'kernel' : model.transformer.kernel,
            'transformer ref_points' : model.transformer.ref_point_head,
            'transformer reference points' : model.transformer.reference_points,
            'transformer init_det' :  model.transformer.init_det, 
            # 'post_process': model.post_process,
            
        }

        # Freeze all parameters
        # for param in model.parameters():
        #     param.requires_grad = False

        # # Unfreeze segmentation head parameters
        # for param in model.PerPixelEmbedding.parameters():
        #     param.requires_grad = True

        # for param in model.AxialBlock.parameters():
        #     param.requires_grad = True
            
        # for param in model.transformer.pos_cross_attention.parameters():
        #     param.requires_grad = True
            
        # for param in model.transformer.parameters():
        #     param.requires_grad = True    

        for name, module in modules.items():
            total_grad_norm = 0
            for param in module.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item()
            writer_grad.add_scalar(f'grad_norm/{name}', total_grad_norm, epoch * len(data_loader) + iteration)

    iteration = 0
    #####################################################################
    # def forward_hook(module, input, output):
    #     # Log the output of each layer (or selected layers) during the forward pass
    #     writer.add_histogram(f'{module.__class__.__name__}_output', output, epoch)
    
    # last_layer = None

    # # Iterate through all modules to find the last one
    # for module in model.modules():
    #     last_layer = module

    # # Check if the last layer is found
    # if last_layer is not None:
    #     # Attach the forward hook to the last layer
    #     last_layer.register_forward_hook(forward_hook)
    #     print(f"Hook attached to the last layer: {last_layer.__class__.__name__}")
    # else:
    #     print("No layer found in the model")
        
        
            
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)
        
        
        loss_dict = criterion(outputs, data_dict)
        weight_dict = criterion.weight_dict
        for key, value in loss_dict.items():
            value.requires_grad_(True)
            
        # print('data_dict is:', data_dict)
        # print('outputs are:', outputs)
        # print('loss_dict is:', loss_dict)
        # print('weight_dict is:', weight_dict)
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses.requires_grad_(True) 
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        
        
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # print('#$#$#$#$#$#$#$#$#$#$losses reduced value in engine.py is:', losses_reduced_scaled)
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        #####################################################################################
        # Store initial weights and biases
        # initial_k_linear_weights = model.module.bbox_attention.k_linear.weight.clone().detach()
        # initial_k_linear_bias = model.module.bbox_attention.k_linear.bias.clone().detach()
        # initial_q_linear_weights = model.module.bbox_attention.q_linear.weight.clone().detach()
        # initial_q_linear_bias = model.module.bbox_attention.q_linear.bias.clone().detach()
        #####################################################################################
        
        optimizer.zero_grad()
        
        # grads = {}
        # def save_grad(name):
        #     def hook(grad):
        #         grads[name] = grad
        #     return hook

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(save_grad(name)) 
        
        losses.backward()
        # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
        #     losses.backward()
        # profiling_results = prof.key_averages().table(sort_by="self_cpu_time_total")
        # output_dir="/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/outputs/gradients.txt"
        # with open (output_dir, 'w') as f:
        #     f.write(profiling_results) 
            
        
        ########################################################################
        # () Logging gradients
        log_gradients(model, writer_grad, epoch, iteration)
        ########################################################################
        
        ################################################### 
        # ()
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm().item()}")  # Print the norm of the gradients
        #     else:
        #         print(f"No gradient or not trainable for {name}")
        
        # for name, grad in grads.items():
        #     if grad is not None:
        #         plt.hist(grad.cpu().numpy().flatten(), bins=100)
        #         plt.title(f'Gradient histogram for {name}')
        #         plt.xlabel('Gradient values')
        #         plt.ylabel('Frequency')
        #         plt.savefig(f'/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR-main-AppleMots/outputs/gradients/gradient_histogram_{name}.png')
        #         plt.close()
        ################################################### 
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        
        ####################################################################################################################
        # Check if weights and biases have changed
        # k_linear_weights_changed = not torch.equal(initial_k_linear_weights, model.module.bbox_attention.k_linear.weight)
        # k_linear_bias_changed = not torch.equal(initial_k_linear_bias, model.module.bbox_attention.k_linear.bias)
        # q_linear_weights_changed = not torch.equal(initial_q_linear_weights, model.module.bbox_attention.q_linear.weight)
        # q_linear_bias_changed = not torch.equal(initial_q_linear_bias, model.module.bbox_attention.q_linear.bias)

        # print(f"k_linear weights changed: {k_linear_weights_changed}")
        # print(f"k_linear bias changed: {k_linear_bias_changed}")
        # print(f"q_linear weights changed: {q_linear_weights_changed}")
        # print(f"q_linear bias changed: {q_linear_bias_changed}")
        ####################################################################################################################

        for key, value in loss_dict.items():
            loss_accumulator[key] += value.item()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes
        iteration+=1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    writer_grad.close()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#########################################################################################################################


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator






