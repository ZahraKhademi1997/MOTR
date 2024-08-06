# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.init import xavier_uniform_
# import io
# from collections import defaultdict
# from typing import Callable, Dict, List, Optional, Tuple, Union
# from util.shapespec import ShapeSpec
# import numpy as np
# from .wrapper import get_norm, Conv2d
# from .wrapper import c2_xavier_fill 


# class FPNEncoder(nn.Module):
#     def __init__(
#         self,
#         # input_shape: List[torch.Size], 
#         input_shape: Dict[str, torch.Size],  
#         conv_dim: int,
#         mask_dim: int,
#         *,
#         norm: Optional[str] = None,  # Assuming 'norm' is a string specifying the type of normalization
#     ):
#         super().__init__()
#         input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
#         self.in_features = [k for k, v in input_shape]
#         # print('self.in_features:', self.in_features)
#         # self.feature_channels = [v.channels for k, v in input_shape]
#         self.feature_channels = [256, 256, 256]
#         self.transformer_feature_strides = [v.stride for k, v in input_shape] 
        
        
#         self.total_num_feature_levels = 4
#         self.common_stride = 4
        
#         self.mask_dim = mask_dim
#         # use 1x1 conv instead
#         # self.mask_features = nn.Conv2d(
#         #     conv_dim,
#         #     mask_dim,
#         #     kernel_size=1,
#         #     stride=1,
#         #     padding=0,
#         # )
#         # nn.init.xavier_uniform_(self.mask_features.weights)
#         # extra fpn levels
#         stride = min(self.transformer_feature_strides)
#         self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 1)
#         print('self.num_fpn_levels:', self.num_fpn_levels)

#         # lateral_convs = []
#         # output_convs = []

#         # use_bias = norm == ""
#         # for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
#         #     lateral_norm = get_norm(norm, conv_dim)
#         #     output_norm = get_norm(norm, conv_dim)

#         #     lateral_conv = Conv2d(
#         #         in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
#         #     )
#         #     output_conv = Conv2d(
#         #         conv_dim,
#         #         conv_dim,
#         #         kernel_size=3,
#         #         stride=1,
#         #         padding=1,
#         #         bias=use_bias,
#         #         norm=output_norm,
#         #         activation=F.relu,
#         #     )
#         #     nn.init.xavier_uniform_(lateral_conv)
#         #     nn.init.xavier_uniform_(output_conv[0].weight)
#         #     self.add_module("adapter_{}".format(idx + 1), lateral_conv)
#         #     self.add_module("layer_{}".format(idx + 1), output_conv)

#         #     lateral_convs.append(lateral_conv)
#         #     output_convs.append(output_conv)
        
#         self.mask_features = nn.Conv2d(
#             conv_dim,
#             mask_dim,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )
#         nn.init.xavier_uniform_(self.mask_features.weight)

        
#         # input_shapes = sorted(enumerate(input_shapes), key=lambda x: x[1].stride)
#         # self.in_features = [i for i, v in input_shapes]
#         # feature_channels = [v.channels for i, v in input_shapes]

#         self.lateral_convs = nn.ModuleList()
#         self.output_convs = nn.ModuleList()

#         for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            
#             lateral_conv = nn.Conv2d(in_channels, conv_dim, kernel_size=1)
#             xavier_uniform_(lateral_conv.weight)
#             self.lateral_convs.append(lateral_conv)

#             output_conv = nn.Sequential(
#                 nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
#                 self._get_norm_layer(norm, conv_dim),
#                 nn.ReLU(inplace=True)
#             )
#             xavier_uniform_(output_conv[0].weight)  # Initialize only the weights of the Conv2d layer
#             self.output_convs.append(output_conv)
            
#         # Place convs into top-down order (from low to high resolution)
#         # to make the top-down computation in forward clearer.
#         self.lateral_convs = self.lateral_convs[::-1]
#         self.output_convs = self.output_convs[::-1]
#         self.high_resolution_index = 0

#     def _get_norm_layer(self, norm_type, num_features):
#         # Define the normalization layer based on the 'norm' argument
#         if norm_type == 'BN':
#             return nn.BatchNorm2d(num_features)
#         elif norm_type == 'GN':
#             # Assuming group number for GroupNorm is 32
#             return nn.GroupNorm(32, num_features)
#         else:
#             # If 'norm' is None or an unrecognized type, return an identity layer
#             return nn.Identity()

        
#     def forward(self, features, srcs):
#         out = []
#         multi_scale_features = []
#         num_cur_levels = 0
#         for i, src in enumerate(srcs):
#             # print('multi_level_memory[i]:', multi_level_memory[i].shape, 'src:', src.shape)
#             reshaped_map = features[i].view(features[i].shape[0] ,features[i].shape[-1],src.shape[-2], src.shape[-1])
#             # print('reshaped_map:', reshaped_map.shape)
#             # Add the reshaped feature map to the new list
#             out.append(reshaped_map)

#         # append `out` with extra FPN levels
#         # Reverse feature maps into top-down order (from low to high resolution)
#         for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
#             x = out[idx].float()
#             lateral_conv = self.lateral_convs[idx]
#             output_conv = self.output_convs[idx]
#             cur_fpn = lateral_conv(x)
#             # Following FPN implementation, we use nearest upsampling here
#             y = cur_fpn + F.interpolate(out[self.high_resolution_index], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            
#             y = output_conv(y)
#             out.append(y)
            
#         for o in out:
            
#             if num_cur_levels < self.total_num_feature_levels:
#                 multi_scale_features.append(o)
#                 num_cur_levels += 1
#         return self.mask_features(out[-1]), out[0], multi_scale_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import io
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union
from util.shapespec import ShapeSpec
import numpy as np
from .wrapper import get_norm, Conv2d
from .wrapper import c2_xavier_fill 


class FPNEncoder(nn.Module):
    def __init__(
        self,
        # input_shape: List[torch.Size], 
        input_shape: Dict[str, torch.Size],  
        conv_dim: int,
        mask_dim: int,
        *,
        norm: Optional[str] = None,  # Assuming 'norm' is a string specifying the type of normalization
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        # print('self.in_features:', self.in_features)
        # self.feature_channels = [v.channels for k, v in input_shape]
        self.feature_channels = [256, 256, 256]
        self.transformer_feature_strides = [v.stride for k, v in input_shape] 
        
        
        self.total_num_feature_levels = 4
        self.common_stride = 4
        
        self.mask_dim = mask_dim
        # use 1x1 conv instead
        # self.mask_features = nn.Conv2d(
        #     conv_dim,
        #     mask_dim,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        # )
        # nn.init.xavier_uniform_(self.mask_features.weights)
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 1)
        print('self.num_fpn_levels:', self.num_fpn_levels)

        # lateral_convs = []
        # output_convs = []

        # use_bias = norm == ""
        # for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
        #     lateral_norm = get_norm(norm, conv_dim)
        #     output_norm = get_norm(norm, conv_dim)

        #     lateral_conv = Conv2d(
        #         in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
        #     )
        #     output_conv = Conv2d(
        #         conv_dim,
        #         conv_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=use_bias,
        #         norm=output_norm,
        #         activation=F.relu,
        #     )
        #     nn.init.xavier_uniform_(lateral_conv)
        #     nn.init.xavier_uniform_(output_conv[0].weight)
        #     self.add_module("adapter_{}".format(idx + 1), lateral_conv)
        #     self.add_module("layer_{}".format(idx + 1), output_conv)

        #     lateral_convs.append(lateral_conv)
        #     output_convs.append(output_conv)
        
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.mask_features.weight)

        
        # input_shapes = sorted(enumerate(input_shapes), key=lambda x: x[1].stride)
        # self.in_features = [i for i, v in input_shapes]
        # feature_channels = [v.channels for i, v in input_shapes]

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            
            lateral_conv = nn.Conv2d(in_channels, conv_dim, kernel_size=1)
            xavier_uniform_(lateral_conv.weight)
            self.lateral_convs.append(lateral_conv)

            output_conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
                self._get_norm_layer(norm, conv_dim),
                nn.ReLU(inplace=True)
            )
            xavier_uniform_(output_conv[0].weight)  # Initialize only the weights of the Conv2d layer
            self.output_convs.append(output_conv)
            
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = self.lateral_convs[::-1]
        self.output_convs = self.output_convs[::-1]
        self.high_resolution_index = 0

    def _get_norm_layer(self, norm_type, num_features):
        # Define the normalization layer based on the 'norm' argument
        if norm_type == 'BN':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'GN':
            # Assuming group number for GroupNorm is 32
            return nn.GroupNorm(32, num_features)
        else:
            # If 'norm' is None or an unrecognized type, return an identity layer
            return nn.Identity()

        
    def forward(self, features, srcs, original_size):
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, src in enumerate(srcs):
            # print('multi_level_memory[i]:', multi_level_memory[i].shape, 'src:', src.shape)
            reshaped_map = features[i].view(features[i].shape[0] ,features[i].shape[-1],src.shape[-2], src.shape[-1])
            # print('reshaped_map:', reshaped_map.shape)
            # Add the reshaped feature map to the new list
            out.append(reshaped_map)
        # for o in out:
        #     print('o', o.shape)
        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = out[idx].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[self.high_resolution_index], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            
            y = output_conv(y)
            out.append(y)
            
        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        final_output = self.mask_features(out[-1])
        # print('final_output:', final_output.shape)
        half_original_size = (original_size[0] // 4, original_size[1] // 4)
        final_output = F.interpolate(final_output, size=half_original_size, mode='bilinear', align_corners=False)

        return final_output, multi_scale_features