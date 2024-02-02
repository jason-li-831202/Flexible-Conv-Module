import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Union, Tuple, Optional

# Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition (https://arxiv.org/pdf/2006.11538)
# CVPR 2020

class PyConv2d(nn.Module):
    """
        PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels for each pyramid level produced by the convolution
        multi_kernels: Spatial size of the kernel for each pyramid level
        multi_groups: Number of blocked connections from input channels to output channels for each pyramid level
        stride: Stride of the convolution. Default: 1
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``False``
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'multi_kernels', 'multi_groups', 
                     'in_channels','out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}
    __split_channels_dict = {
        1: [1], 
        2: [2]*2,
        3: [4, 4, 2],
        4: [4]*4
    }
    def __init__(self,  in_channels: int, 
                        out_channels:int , 
                        multi_kernels: list, 
                        multi_groups: list, 
                        stride: int = 1,
                        dilation: int = 1, 
                        bias: bool  = False, 
                        padding_mode: str = 'zeros'):
        super(PyConv2d, self).__init__()

        if len(multi_kernels) != len(multi_groups):
            raise ValueError('Num of kernels and groups list not match!')
        elif len(multi_kernels) > 4:
            raise ValueError("Num of kernels cannot exceed 4!")
        
        levels = []
        split_ratio = self.__split_channels_dict[len(multi_kernels)]
        
        for ratio, kernel, group in zip(split_ratio, multi_kernels, multi_groups):
            stride = _pair(stride)
            kernel = _pair(kernel)
            padding =  (dilation*(kernel[0]-1)//2, dilation*(kernel[1]-1)//2)

            level = nn.Conv2d(in_channels, out_channels // ratio, kernel_size=kernel,
                            stride=stride, padding=padding, groups=group,
                            dilation=dilation, bias=bias, padding_mode=padding_mode)
            levels.append(level)
        self.pyConv = nn.ModuleList(levels)

    def forward(self, x):
        out = []
        for level in self.pyConv:
            out.append(level(x))

        return torch.cat(out, dim=1)
