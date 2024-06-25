import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import io
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union
from util.shapespec import ShapeSpec
    

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


    def forward(self, features, original_size):
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
        half_original_size = (original_size[0] // 4, original_size[1] // 4)
        final_output = F.interpolate(final_output, size=half_original_size, mode='bilinear', align_corners=False)
        return final_output, out_features
    

    
