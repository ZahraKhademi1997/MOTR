# Copyright (c) Hikvision Research Institute. All rights reserved.
import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
import torch.nn.init as init


class DynamicDeformableAttention(BaseModule):
    """A dynamic attention module used in SOIT. The parameters of this module
    are generated from transformer decoder head.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0.1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=1,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.im2col_step_test = 100  # for test
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        # Initialize weights and biases
        self.sampling_offsets_weight = nn.Parameter(torch.Tensor(256, 8))
        self.sampling_offsets_bias = nn.Parameter(torch.Tensor(32))
        self.attention_weights_weight = nn.Parameter(torch.Tensor(128, 8))
        self.attention_weights_bias = nn.Parameter(torch.Tensor(16))
        self.output_proj_weights = nn.Parameter(torch.Tensor(8, 1))
        self.output_proj_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(self.sampling_offsets_weight)
        init.xavier_uniform_(self.attention_weights_weight)
        init.xavier_uniform_(self.output_proj_weights)

        # Initialize biases
        init.constant_(self.sampling_offsets_bias, 0)
        init.constant_(self.attention_weights_bias, 0)
        init.constant_(self.output_proj_bias, 0)
        
    def forward(self,
                dynamic_params,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            dynamic_params (Tensor): Dynamic generated parameters for 
                MultiScaleDeformAttention with shape (dynamic_params_dims).
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_key

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_key, self.num_heads, -1)
        #split the dynamic parameters to each layer
        sampling_offsets_weight = dynamic_params[:256].reshape(32, 8)
        sampling_offsets_bias = dynamic_params[256:288]
        attention_weights_weight = dynamic_params[288:416].reshape(16, 8)
        attention_weights_bias = dynamic_params[416:432]
        output_proj_weights = dynamic_params[432:440].reshape(1, 8)
        output_proj_bias = dynamic_params[440]
        sampling_offsets = F.linear(
            query, sampling_offsets_weight, sampling_offsets_bias).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = F.linear(
            query, attention_weights_weight, attention_weights_bias).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
 
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)

        output = output.relu()     
        output = F.linear(
            output, output_proj_weights, output_proj_bias).permute(1, 0, 2)
        return output

    def forward_test(self,
                     dynamic_params,
                     query,
                     key,
                     value,
                     residual=None,
                     query_pos=None,
                     key_padding_mask=None,
                     reference_points=None,
                     spatial_shapes=None,
                     level_start_index=None,
                     **kwargs):
        """Faster version for dynamic encoder inference"""

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_key

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_key, self.num_heads, -1)

        query = query.permute(0, 2, 1).reshape(1, -1, num_query)
        sampling_offsets_weight = dynamic_params[:, :256].reshape(bs * 32, 8, 1)
        sampling_offsets_bias = dynamic_params[:, 256:288].reshape(bs * 32)
        attention_weights_weight = dynamic_params[:, 288:416].reshape(bs * 16, 8, 1)
        attention_weights_bias = dynamic_params[:, 416:432].reshape(bs * 16)
        output_proj_weights = dynamic_params[:, 432:440].reshape(bs * 1, 8, 1)
        output_proj_bias = dynamic_params[:, 440].reshape(bs * 1)
        sampling_offsets = F.conv1d(
            query, sampling_offsets_weight, sampling_offsets_bias, \
                groups=bs).view(bs, self.num_heads, self.num_levels, \
                    self.num_points, 2, num_query).permute(
                        0, 5, 1, 2, 3, 4).contiguous()
        attention_weights = F.conv1d(
            query, attention_weights_weight, attention_weights_bias, \
                groups=bs).view(bs, self.num_heads, \
                    self.num_levels * self.num_points, num_query).permute(
                        0, 3, 1, 2).contiguous()

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step_test)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step_test)
        output = output.relu()

        output = output.permute(0, 2, 1).reshape(1, -1, num_query)
        output = F.conv1d(output,
                          output_proj_weights,
                          output_proj_bias,
                          groups=bs).permute(1, 0, 2)
        return output


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]
