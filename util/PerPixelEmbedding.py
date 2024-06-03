import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import io
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union
from util.shapespec import ShapeSpec
    
# FPN structure   
# class PerPixelEmbedding(nn.Module):
    
#     def __init__(
#         self,
#         # input_shape: Dict[str, ShapeSpec],  
#         input_shape: Dict[str, torch.Size], 
#         conv_dim: int,
#         mask_dim: int,
#         *,
#         norm: Optional[Union[str, Callable]] = None
#     ):
#         super().__init__()
        
#         input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride, reverse=True)
#         self.in_features = [k for k, v in input_shape]
#         feature_channels = [v.channels for k, v in input_shape]

#         self.lateral_convs = nn.ModuleList()
#         self.output_convs = nn.ModuleList()
        
#         for idx, in_channels in enumerate(feature_channels):
#             if idx < len(self.in_features) - 1:  # Intermediate layers
#                 lateral_conv = nn.Conv2d(in_channels, conv_dim, kernel_size=1)
#                 xavier_uniform_(lateral_conv.weight)
#                 self.lateral_convs.append(lateral_conv)

#                 output_conv = nn.Sequential(
#                     nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
#                     self._get_norm_layer(norm, conv_dim),
#                     nn.ReLU(inplace=True)
#                 )
#                 xavier_uniform_(output_conv[0].weight)  # Initialize only the weights of the Conv2d layer
#                 self.output_convs.append(output_conv)
#             else:  # Last layer
#                 output_conv = nn.Sequential(
#                     nn.Conv2d(in_channels, mask_dim, kernel_size=3, stride=1, padding=1),
#                     self._get_norm_layer(norm, mask_dim),
#                     nn.ReLU(inplace=True)
#                 )
#                 xavier_uniform_(output_conv[0].weight)  # Initialize only the weights of the Conv2d layer
#                 self.output_convs.append(output_conv)
#                 # No lateral convolutions for the last layer
#                 self.lateral_convs.append(None)

#         self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.mask_features.weight)
#         self.num_feature_levels = 3
        
#     def _get_norm_layer(self, norm_type, num_features):
#         if norm_type == 'BN':
#             return nn.BatchNorm2d(num_features)
#         elif norm_type == 'GN':
#             return nn.GroupNorm(32, num_features)
#         else:
#             return nn.Identity()
        
#     def forward(self, features):
#         prev_feature = None
#         out_features = []
#         num_cur_levels = 0
#         print('self.in_features:', self.in_features)
#         # Process features in a top-down manner (from high to low resolution)
#         for idx in range(len(self.in_features)-1, -1, -1):
#             x = features[idx].tensors
#             feature_key = self.in_features[idx]
#             print('feature_key:', feature_key)
#             lateral_conv = self.lateral_convs[idx]
#             output_conv = self.output_convs[idx]
#             # print('features:', features)

#             # Apply lateral convolution if available
#             current_feature = lateral_conv(x) if lateral_conv else x

#             if prev_feature is not None:
#                 # Upsample the previous feature to the current feature's size and add
#                 upsampled_feature = F.interpolate(prev_feature, size=current_feature.shape[-2:], mode='nearest')
#                 current_feature = current_feature + upsampled_feature

#             # Apply output convolution
#             current_feature = output_conv(current_feature)

#             if num_cur_levels < self.num_feature_levels:
#                 out_features.append(current_feature)
#                 num_cur_levels += 1

#             prev_feature = current_feature

#         # Final convolution on the last feature map
#         mask_features = self.mask_features(prev_feature)
#         return mask_features, out_features

# UP_BOTTOM FPN PATH
class PerPixelEmbedding(nn.Module):
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

        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.mask_features.weight)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]
        
        # input_shapes = sorted(enumerate(input_shapes), key=lambda x: x[1].stride)
        # self.in_features = [i for i, v in input_shapes]
        # feature_channels = [v.channels for i, v in input_shapes]

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for idx, in_channels in enumerate(feature_channels):
            if idx < len(self.in_features) - 1:  # Intermediate layers
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
            else:  # Last layer
                output_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mask_dim, kernel_size=3, stride=1, padding=1),
                    self._get_norm_layer(norm, mask_dim),
                    nn.ReLU(inplace=True)
                )
                xavier_uniform_(output_conv[0].weight)  # Initialize only the weights of the Conv2d layer
                self.output_convs.append(output_conv)
                # No lateral convolutions for the last layer
                self.lateral_convs.append(None)
        self.num_feature_levels = 3

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


    def forward(self, features):
        prev_feature = None
        out_features = []
        num_cur_levels = 0

        # Iterate through features in reverse order to implement the top-down pathway
        for idx in range(len(self.in_features)-1, -1, -1):
            x = features[idx].tensors  # Assuming features is a list of tensors
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]

            if lateral_conv is not None:
                x = lateral_conv(x)  # Apply the lateral convolution

            if prev_feature is not None:
                # Upsample the previous feature to match the size of the current feature
                upsampled_feature = F.interpolate(prev_feature, size=x.shape[-2:], mode='nearest')
                x += upsampled_feature  # Add upsampled previous features to the current features

            current_feature = output_conv(x)  # Apply the output convolution to refine the features

            if num_cur_levels < self.num_feature_levels:
                out_features.append(current_feature)  # Collect the processed features
                num_cur_levels += 1

            prev_feature = current_feature  # Set the current feature as the previous for the next iteration

        # Apply a final convolution to the last processed feature map
        final_output = self.mask_features(prev_feature)

        return final_output, out_features
    

    # def forward(self, features):
    #     y = None
    #     num_cur_levels = 0
    #     out_features = []
    #     for idx in range(len(self.in_features)-1, -1, -1):  # Iterate backwards over the indices
    #         # print('idx:', idx)
    #         x = features[idx].tensors # Nested tensor
    #         # print('x:', x)
    #         # print('x.shape:', x.shape)
    #         lateral_conv = self.lateral_convs[idx]
    #         output_conv = self.output_convs[idx]
    #         if lateral_conv is not None:
    #             x = lateral_conv(x)
    #         if y is not None:
    #             y = F.interpolate(y, size=x.shape[-2:], mode='nearest')
    #         y = x if y is None else x + y
    #         y = output_conv(y)
    #     return self.mask_features(y)



# # BOTTOM_UP FPN PATH
# class PerPixelEmbedding(nn.Module):
#     def __init__(
#         self,
#         input_shape: Dict[str, torch.Size],  
#         conv_dim: int,
#         mask_dim: int,
#         *,
#         norm: Optional[str] = None,
#     ):
#         super().__init__()

#         self.mask_dim = mask_dim
#         self.mask_features = nn.Conv2d(
#             conv_dim,
#             mask_dim,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )
#         nn.init.xavier_uniform_(self.mask_features.weight)

#         # Sort input shapes by stride in ascending order (reverse of the traditional FPN)
#         input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride, reverse=False)
#         self.in_features = [k for k, v in input_shape]
#         feature_channels = [v.channels for k, v in input_shape]

#         self.lateral_convs = nn.ModuleList()
#         self.output_convs = nn.ModuleList()

#         for idx, in_channels in enumerate(feature_channels):
#             # Every level gets a lateral conv except the first (smallest feature map)
#             if idx > 0:
#                 lateral_conv = nn.Conv2d(in_channels, conv_dim, kernel_size=1)
#                 nn.init.xavier_uniform_(lateral_conv.weight)
#                 self.lateral_convs.append(lateral_conv)
#             else:
#                 self.lateral_convs.append(None)  # No lateral conv for the first layer
            
#             # Output convolutions for all levels
#             output_conv = nn.Sequential(
#                 nn.Conv2d(conv_dim if idx > 0 else in_channels, conv_dim, kernel_size=3, stride=1, padding=1),
#                 self._get_norm_layer(norm, conv_dim),
#                 nn.ReLU(inplace=True)
#             )
#             nn.init.xavier_uniform_(output_conv[0].weight)
#             self.output_convs.append(output_conv)

#     def _get_norm_layer(self, norm_type, num_features):
#         if norm_type == 'BN':
#             return nn.BatchNorm2d(num_features)
#         elif norm_type == 'GN':
#             return nn.GroupNorm(32, num_features)
#         else:
#             return nn.Identity()

#     def forward(self, features):
#         prev_feature = None
#         out_features = []

#         # Iterate through features in ascending order
#         for idx in range(len(self.in_features)):
#             x = features[idx].tensors  # Assuming features is a list of tensors
#             lateral_conv = self.lateral_convs[idx]
#             output_conv = self.output_convs[idx]

#             if lateral_conv is not None:
#                 x = lateral_conv(x)  # Apply the lateral convolution

#             if prev_feature is not None:
#                 # Upsample the previous feature to match the size of the current feature
#                 prev_feature = F.interpolate(prev_feature, size=x.shape[-2:], mode='nearest')
#                 x += prev_feature  # Add upsampled previous features to the current features

#             current_feature = output_conv(x)  # Apply the output convolution to refine the features
#             out_features.append(current_feature)
#             prev_feature = current_feature  # Set the current feature as the previous for the next iteration

#         # Apply a final convolution to the last processed feature map
#         final_output = self.mask_features(prev_feature)

#         return final_output, out_features

