# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter, UninitializedParameter
# import torch.nn.init as init


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class LocalEmbeddingUnit(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_depthwise=False):
#         super(LocalEmbeddingUnit, self).__init__()
#         if use_depthwise:
#             # Depthwise convolution
#             self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
#             self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         else:
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)

#     def forward(self, x):
#         x = self.conv(x)
#         if hasattr(self, 'pointwise'):
#             x = self.pointwise(x)
#         return x
        
# # class qkv_transform(nn.Conv1d):
# #     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
# #         stride = stride if isinstance(stride, int) else stride[0]
# #         stride = int(1)
# #         print('stride:', stride)
        
# #         super(qkv_transform, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias)
# #         self.weight.data.normal_(0, math.sqrt(1. / in_planes))
        
# class qkv_transform(nn.Conv1d):
#     def __init__(self, in_planes, out_planes, groups = 8, kernel_size=1, stride=1, padding=0, bias=False):
#         kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
#         stride = (stride,) if isinstance(stride, int) else stride
#         padding = (padding,) if isinstance(padding, int) else padding
#         dilation = 1
#         super(qkv_transform, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias)

        
#         # self.weight = Parameter(torch.Tensor(out_planes, in_planes // groups, *kernel_size))
#         super(qkv_transform, self).__init__(
#             in_channels=in_planes, 
#             out_channels=out_planes, 
#             kernel_size=kernel_size, 
#             stride=stride, 
#             padding=padding, 
#             dilation=dilation,  # Ensuring dilation is passed correctly
#             groups=groups, 
#             bias=bias
#         )
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_planes))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters_qkv()

#     def reset_parameters_qkv(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
#         # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             if fan_in != 0:
#                 bound = 1 / math.sqrt(fan_in)
#                 init.uniform_(self.bias, -bound, bound)


# class AxialAttention(nn.Module):
#     # def __init__(self, in_planes, out_planes, groups=8, kernel_size=32,
#     #              stride=1, bias=False, width=False):
#     def __init__(self, in_planes, out_planes, groups=8, base_kernel_size = 200,
#                  stride=1, bias=False, width=False):
#         assert (in_planes % groups == 0) and (out_planes % groups == 0)
#         super(AxialAttention, self).__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.groups = groups
#         self.group_planes = out_planes // groups
#         self.base_kernel_size = base_kernel_size
#         self.stride = stride
#         self.bias = bias
#         # self.stride = stride if isinstance(stride, tuple) else (stride,)
#         # self.stride = stride if isinstance(stride, int) else stride[0]
#         # self.padding = (0,) if isinstance(padding, int) else padding
#         self.width = width


#         # Multi-head self attention
#         self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
#         self.bn_similarity = nn.BatchNorm2d(groups * 3)
#         self.bn_output = nn.BatchNorm1d(out_planes * 2)
#         # Position embedding
#         self.relative = nn.Parameter(torch.randn(self.group_planes * 2, base_kernel_size * 2 - 1), requires_grad=True)
        
#         if stride > 1:
#             self.pooling = nn.AvgPool2d(stride, stride=stride)

#         self.reset_parameters()

#     def forward(self, x):
#         device = x.device  # Get the device from input tensor
#         # self.relative.to(device)
#         self.relative_col.to(device)
#         self.relative_row.to(device)
    
#         if self.width:
#             x = x.permute(0, 2, 1, 3)
#         else:
#             x = x.permute(0, 3, 1, 2)  # N, W, C, H
#         N, W, C, H = x.shape
#         # print('x shape before:', x.shape)
#         x = x.contiguous().view(N * W, C, H)
#         # print('x shape:', x.shape) #torch.Size([134, 128, 96])
        
#         # kernel_size = x.shape[2]
#         # R = x.shape[2] if not self.width else x.shape[0]
#         R = x.shape[2] 
#         target_size = R * 2 - 1
#         current_relative = F.interpolate(self.relative.unsqueeze(0), size=target_size, mode='nearest').squeeze(0).to(device)
#         query_index = torch.arange(R).unsqueeze(0)
#         key_index = torch.arange(R).unsqueeze(1)
#         relative_index = key_index - query_index + R - 1
#         flatten_index = relative_index.view(-1).to(device)

#         # Transformations
#         qkv_before = self.qkv_transform(x)
#         # print('qkv_before:', qkv_before.shape) #qkv_before: torch.Size([134, 256, 96])
#         qkv = self.bn_qkv(qkv_before)
#         q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

#         # Calculate position embedding
#         # all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size) 
#         all_embeddings = torch.index_select(current_relative, 1, flatten_index).view(self.group_planes * 2, R, R)
#         # print('all embedding:', all_embeddings.shape) # torch.Size([32, 123, 123])
#         q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
#         # print('q:', q.shape, 'q_embedding:', q_embedding.shape) #  q: torch.Size([134, 8, 8, 96]), q_embedding: torch.Size([8, 123, 123])
#         # print('v:', v.shape, 'v_embedding:', v_embedding.shape) # v: torch.Size([134, 8, 16, 96]) v_embedding: torch.Size([16, 123, 123])
#         # print('k:', k.shape, 'k_embedding:', k_embedding.shape) # k: torch.Size([134, 8, 8, 96]) k_embedding: torch.Size([8, 123, 123])

#         qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
#         kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
#         qk = torch.einsum('bgci, bgcj->bgij', q, k)
#         stacked_similarity = torch.cat([qk, qr, kr], dim=1)
#         stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
#         #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
#         # (N, groups, H, H, W)
#         similarity = F.softmax(stacked_similarity, dim=3)
#         if self.width:
#             sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
#         sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

#         if self.width:
#             stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
#         output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

#         if self.width:
#             output = output.permute(0, 2, 1, 3)
#         else:
#             output = output.permute(0, 2, 3, 1)

#         if self.stride > 1:
#             output = self.pooling(output)

#         return output , similarity

#     def reset_parameters(self):
#         # self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
#         #nn.init.uniform_(self.relative, -0.1, 0.1)
#         nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))



# class AxialBlock(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=8,
#                 base_width=32, dilation=1, norm_layer=None, kernel_size=200):

#         super(AxialBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 32.))
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         # Initialize local embedding unit
#         self.local_embed = LocalEmbeddingUnit(inplanes, width, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        
#         self.conv_down = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.hight_block = AxialAttention(width, width, groups=groups, base_kernel_size=kernel_size)
#         self.width_block = AxialAttention(width, width, groups=groups, base_kernel_size=kernel_size, stride=stride, width=True)
#         self.conv_up = conv1x1(width, planes * self.expansion)
#         self.bn2 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
        
#         x = self.local_embed(x)
#         # identity = x

#         out = self.conv_down(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out_row, similarity_w = self.width_block(out)
#         out_col, similarity_h = self.hight_block(out)
        
#         out = self.relu(out)

#         out = self.conv_up(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out , similarity_h, similarity_w



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.init as init


# class qkv_transform(nn.Conv1d):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
#         stride = stride if isinstance(stride, int) else stride[0]
#         stride = int(1)
#         print('stride:', stride)
        
#         super(qkv_transform, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias)
#         self.weight.data.normal_(0, math.sqrt(1. / in_planes))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        
class qkv_transform(nn.Conv1d):
    def __init__(self, in_planes, out_planes, groups = 8, kernel_size=1, stride=1, padding=0, bias=False,  device=None):
        kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        stride = (stride,) if isinstance(stride, int) else stride
        padding = (padding,) if isinstance(padding, int) else padding
        dilation = 1
        super(qkv_transform, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias)

        
        # self.weight = Parameter(torch.Tensor(out_planes, in_planes // groups, *kernel_size))
        super(qkv_transform, self).__init__(
            in_channels=in_planes, 
            out_channels=out_planes, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,  # Ensuring dilation is passed correctly
            groups=groups, 
            bias=bias,
            device=device
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_planes))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters_qkv()

    def reset_parameters_qkv(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

class AxialAttention(nn.Module):
    # def __init__(self, in_planes, out_planes, groups=8, kernel_size=32,
    #              stride=1, bias=False, width=False):
    def __init__(self, in_planes, out_planes, groups=8, base_kernel_size = 200,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.base_kernel_size = base_kernel_size
        self.stride = stride
        self.bias = bias
        # self.stride = stride if isinstance(stride, tuple) else (stride,)
        # self.stride = stride if isinstance(stride, int) else stride[0]
        # self.padding = (0,) if isinstance(padding, int) else padding
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, base_kernel_size * 2 - 1), requires_grad=True)
        # query_index = torch.arange(kernel_size).unsqueeze(0)
        # key_index = torch.arange(kernel_size).unsqueeze(1)
        # relative_index = key_index - query_index + kernel_size - 1
        # self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        device = x.device  # Get the device from input tensor
        self.relative.to(device)

        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        # print('x shape before:', x.shape)
        x = x.contiguous().view(N * W, C, H)
        # print('x shape:', x.shape) #torch.Size([134, 128, 96])
        
        # kernel_size = x.shape[2]
        # R = x.shape[2] if not self.width else x.shape[0]
        R = x.shape[2] 
        target_size = R * 2 - 1
        current_relative = F.interpolate(self.relative.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
        query_index = torch.arange(R).unsqueeze(0)
        key_index = torch.arange(R).unsqueeze(1)
        relative_index = key_index - query_index + R - 1
        flatten_index = relative_index.view(-1).to(device)

        # Transformations
        qkv_before = self.qkv_transform(x)
        # print('qkv_before:', qkv_before.shape) #qkv_before: torch.Size([134, 256, 96])
        qkv = self.bn_qkv(qkv_before)
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        # all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size) 
        all_embeddings = torch.index_select(current_relative, 1, flatten_index).view(self.group_planes * 2, R, R)
        # print('all embedding:', all_embeddings.shape) # torch.Size([32, 123, 123])
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        # print('q:', q.shape, 'q_embedding:', q_embedding.shape) #  q: torch.Size([134, 8, 8, 96]), q_embedding: torch.Size([8, 123, 123])
        # print('v:', v.shape, 'v_embedding:', v_embedding.shape) # v: torch.Size([134, 8, 16, 96]) v_embedding: torch.Size([16, 123, 123])
        # print('k:', k.shape, 'k_embedding:', k_embedding.shape) # k: torch.Size([134, 8, 8, 96]) k_embedding: torch.Size([8, 123, 123])

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        # return output , similarity
        return output 

    def reset_parameters(self):
        # self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))



class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=8,
                base_width=32, dilation=1, norm_layer=None, kernel_size=200):

        super(AxialBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = lambda num_features: nn.BatchNorm2d(num_features)
        width = int(planes * (base_width / 32.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, base_kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, base_kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        if self.downsample is not None:
            self.downsample = self.downsample 

    def forward(self, x):
        # x = x.to('cuda:7')
        identity = x
        # if x.device != self.conv_down.weight.device:
        #     print(f"Device mismatch: input on {x.device}, conv_down on {self.conv_down.weight.device}")

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out, similarity_h = self.hight_block(out)
        # out, similarity_w = self.width_block(out)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # return out , similarity_h, similarity_w
        return out 
