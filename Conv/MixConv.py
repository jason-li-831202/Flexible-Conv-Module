import torch
import numpy as np
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict


__all__ = ["MixedDepthwiseConv2d"]
# MixConv: Mixed Depthwise Convolutional Kernels (https://arxiv.org/abs/1907.09595)
# BMVC 2019

class MixedDepthwiseConv2d(nn.Module):
    """
        Use convolution kernels of different sizes to group different channels of depth convolution. 
        It can also be regarded as a mixture of multiple convolution kernels in the Depthwise + Inception structure.
    """
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        multi_kernels: list = [3, 5, 7], 
                        stride: int = 1, 
                        dilation: int = 1,
                        bias: bool = False, 
                        padding_mode: str = 'zeros',
                        split_mode: str = "equal"):
        """
        in_channels: Number of channels in the input image
        out_channels: Number of channels for each pyramid level produced by the convolution
        multi_kernels: Spatial size of the kernel for each pyramid level
        stride: Stride of the convolution. Default: 1
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: False
        equal_ch: two ways of channel division: equal and exp. Default: equal
        """
        super(MixedDepthwiseConv2d, self).__init__()
        self.split_mode = split_mode
        stride = _pair(stride)
        multi_channels = self._split_channels(out_channels, multi_kernels)

        levels = []
        for split_chaneel, kernel in zip(multi_channels, multi_kernels):
                kernel = _pair(kernel)
                padding = (dilation*(kernel[0]-1)//2, dilation*(kernel[1]-1)//2)

                level = nn.Conv2d(in_channels, int(split_chaneel), kernel_size=kernel,
                                    stride=stride, padding=padding, groups=1,
                                    dilation=dilation, bias=bias, padding_mode=padding_mode)
                levels.append(level)
                self.__setattr__('mixConv{}x{}'.format(*kernel), level)
        self.mixConv = nn.ModuleList(levels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.identity = nn.Identity() if (out_channels == in_channels and stride == (1, 1)) else None
        print('MixConv, identity = ', self.identity)

    def _split_channels(self, channel, k):		# 根据组数对输入通道分组
        groups = len(k)
        if self.split_mode == "exp": 
            # 指数划分通道
            b = [channel] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b
        else: 
            # 均等划分通道
            i = torch.linspace(0, groups - 1E-6, channel).floor()
            c_ = [(i == g).sum() for g in range(groups)] 
        return c_

    def forward(self, x):
        out = self.act(self.bn(torch.cat([m(x) for m in self.mixConv], dim=1)))
        if (self.identity is not None) :
            out += self.identity(x)
        return out
