# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List
import numpy as np
from util import box_ops
from util.misc import inverse_sigmoid
from models.structures import Boxes, Instances, pairwise_iou
import gc
# from memory_profiler import profile

# ######################################################################################
# # (1) Memory monitoring
# import sys
# # def print_gpu_memory():
# #     print(torch.cuda.memory_summary(device=None, abbreviated=False))
# def log_gpu_memory(file_path, message):
#     # print(f"Logging GPU memory to {file_path} - {message}")
#     original_stdout = sys.stdout
#     with open(file_path, "a") as f:  # Use 'a' to append to the file instead of overwriting it
#         sys.stdout = f
#         print(message)
#         print(torch.cuda.memory_summary(device=None, abbreviated=False))
#     sys.stdout = original_stdout
# ######################################################################################

def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        
        if self.training:
            ##################################################################################
            # (2) Solving the device problem
            device = 'cuda:0'
            track_instances = track_instances.to(device)
            ##################################################################################
            
            ##################################################################################
            # (3) changing track_instances.iou to track_instances.iou_boxes --> motr.py
            # active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou_boxes > 0.5)
            ##################################################################################
            
            active_idxes = active_idxes.to(device)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            track_instances = track_instances.to(track_instances.obj_idxes.device)
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    # @profile
    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']

        #######################################################################################################################################################################################
        # (4)
        # log_gpu_memory("/home/zahra/Documents/Projects/prototype/MOTR/new/gpu_memory_log.txt", "Before interpolation in motr.py:")
        processed_active_masks = []
        if len(processed_active_masks) > 0:
            print(' Removing stacked list')
            del processed_active_masks
            gc.collect()
        
        
        for i in range(len(active_track_instances)):
            active_track_mask = active_track_instances[i].get("pred_masks") 
            init_track_mask = init_track_instances[i].get("pred_masks")
                
            if (active_track_mask.shape[1] != init_track_mask.shape[1]) and (active_track_mask.shape[2] != init_track_mask.shape[2]):
                print("active_track_mask for dim1 and dim2 before has the shape of:", active_track_mask.shape)
                print("init_track_mask for dim1 and dim2 before has the shape of:", init_track_mask.shape)
                if active_track_mask.dim() != 2:
                    print('active_track_mask in qim is not three dim:', active_track_mask.shape)
                    # Ensure active_track_mask has three dimensions
                    active_track_mask = active_track_mask.squeeze(0)

                if init_track_mask.dim() != 2:
                    print('init_track_mask in qim is not three dim:', init_track_mask.shape)
                    # Ensure init_track_mask has three dimensions
                    init_track_mask = init_track_mask.squeeze(0)
                
                # Interpolate along dimension 2 (width) to match init_track_mask width
                target_size = (init_track_mask.shape[0], init_track_mask.shape[1])

                active_track_mask = torch.nn.functional.interpolate(
                    active_track_mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                if active_track_mask.dim() != 3:
                    print('active_track_mask in qim is not three dim:', active_track_mask.shape)
                    # Ensure active_track_mask has three dimensions
                    active_track_mask = active_track_mask.squeeze(0)

                if init_track_mask.dim() != 3:
                    print('init_track_mask in qim is not three dim:', init_track_mask.shape)
                    # Ensure init_track_mask has three dimensions
                    init_track_mask = init_track_mask.unsqueeze(0)
                
                # active_track_mask = active_track_mask.squeeze() 
                print("active_track_mask for dim1 and dim2 after has the shape of:", active_track_mask.shape)
                print("init_track_mask for dim1 and dim2 after has the shape of:", init_track_mask.shape)
                processed_active_masks.append(active_track_mask.cpu().numpy())
            
            elif active_track_mask.shape[1] != init_track_mask.shape[1]:
                print("active_track_mask for dim1 before has the shape of:", active_track_mask.shape)
                print("init_track_mask for dim1 before has the shape of:", init_track_mask.shape)
                if active_track_mask.dim() != 2:
                    print('active_track_mask in qim is not three dim:', active_track_mask.shape)
                    # Ensure active_track_mask has three dimensions
                    active_track_mask = active_track_mask.squeeze(0)

                if init_track_mask.dim() != 2:
                    print('init_track_mask in qim is not three dim:', init_track_mask.shape)
                    # Ensure init_track_mask has three dimensions
                    init_track_mask = init_track_mask.squeeze(0)
                
                # Interpolate along dimension 2 (width) to match init_track_mask width
                target_size = (init_track_mask.shape[0], active_track_mask.shape[1])

                active_track_mask = torch.nn.functional.interpolate(
                    active_track_mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                if active_track_mask.dim() != 3:
                    print('active_track_mask in qim is not three dim:', active_track_mask.shape)
                    # Ensure active_track_mask has three dimensions
                    active_track_mask = active_track_mask.squeeze(0)

                if init_track_mask.dim() != 3:
                    print('init_track_mask in qim is not three dim:', init_track_mask.shape)
                    # Ensure init_track_mask has three dimensions
                    init_track_mask = init_track_mask.unsqueeze(0)
                
                # active_track_mask = active_track_mask.squeeze() 
                print("active_track_mask for dim1 after has the shape of:", active_track_mask.shape)
                print("init_track_mask for dim1 after has the shape of:", init_track_mask.shape)
                processed_active_masks.append(active_track_mask.cpu().numpy())
                
                   
            elif active_track_mask.shape[2] != init_track_mask.shape[2]:
                print("active_track_mask for dim2 before has the shape of:", active_track_mask.shape)
                print("init_track_mask for dim2 before has the shape of:", init_track_mask.shape)
                if active_track_mask.dim() != 2:
                    print('active_track_mask in qim is not three dim:', active_track_mask.shape)
                    # Ensure active_track_mask has three dimensions
                    active_track_mask = active_track_mask.squeeze(0)

                if init_track_mask.dim() != 2:
                    print('init_track_mask in qim is not three dim:', init_track_mask.shape)
                    # Ensure init_track_mask has three dimensions
                    init_track_mask = init_track_mask.squeeze(0)
                
                # Interpolate along dimension 2 (width) to match init_track_mask width
                target_size = (active_track_mask.shape[0], init_track_mask.shape[1])

                active_track_mask = torch.nn.functional.interpolate(
                    active_track_mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                if active_track_mask.dim() != 3:
                    print('active_track_mask in qim is not three dim:', active_track_mask.shape)
                    # Ensure active_track_mask has three dimensions
                    active_track_mask = active_track_mask.squeeze(0)

                if init_track_mask.dim() != 3:
                    print('init_track_mask in qim is not three dim:', init_track_mask.shape)
                    # Ensure init_track_mask has three dimensions
                    init_track_mask = init_track_mask.unsqueeze(0)
                
                # active_track_mask = active_track_mask.squeeze() 
                print("active_track_mask for dim2 after has the shape of:", active_track_mask.shape)
                print("init_track_mask for dim2 after has the shape of:", init_track_mask.shape)
                processed_active_masks.append(active_track_mask.cpu().numpy())
        
          
        if len(processed_active_masks) > 0:
            # Check the type of each item in the list and verify the structure
            # for mask in processed_active_masks:
            #     if isinstance(mask, torch.Tensor):
            #         # Access the data within the tensor using mask
            #         processed_active_tensors = processed_active_masks
            #         break  # Exit the loop when you find a PyTorch tensor
            # else:
            #     print("No PyTorch tensors found in processed_active_masks.")
            #     processed_active_tensors = None
                   
            processed_active_tensors = [mask for mask in processed_active_masks]
            processed_active_masks = np.concatenate(processed_active_tensors, axis=0)
            print(" processed_active_masks has the shape of:",  processed_active_masks.shape)
            processed_active_masks = torch.from_numpy(processed_active_masks).to("cuda:0")
            processed_active_masks = processed_active_masks.squeeze(1)
            
            
            # Update the pred_masks field in active_track_instances with processed masks
            active_track_instances.set("pred_masks", processed_active_masks)
            print("Size of updated pred_masks in active_track_instances:", active_track_instances.pred_masks.shape)
            # Now, define processed_active_instances
            processed_active_instances = active_track_instances
           
        else:
            processed_active_instances = active_track_instances
        # log_gpu_memory("/home/zahra/Documents/Projects/prototype/MOTR/new/gpu_memory_log.txt", "After interpolation in motr.py:")
        #######################################################################################################################################################################################
        merged_track_instances = Instances.cat([init_track_instances, processed_active_instances])
        return merged_track_instances


def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'QIM': QueryInteractionModule,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
