import functools
import warnings
from typing import List, Optional
import torch
from torch.nn import functional as F
import torch.nn as nn


BatchNorm2d = torch.nn.BatchNorm2d

    
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def get_norm(norm, out_channels):
        """
        Args:
            norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
                or a callable that takes a channel number and returns
                the normalization layer as a nn.Module.

        Returns:
            nn.Module or None: the normalization layer
        """
        if norm is None:
            return None
        if isinstance(norm, str):
            if len(norm) == 0:
                return None
            norm = {
                "BN": BatchNorm2d,
                #
                "GN": lambda channels: nn.GroupNorm(32, channels),
                
                "nnSyncBN": nn.SyncBatchNorm,
                
            }[norm]
        return norm(out_channels)
    
def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)