import torch
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict
from torch.nn import functional as F

from Conv.Basic.common import ConvBN

__all__ = ["GhostConv2d", "Ghostv2Conv2d"]
# GhostNet: More Features from Cheap Operations (​​​​​​https://arxiv.org/abs/1911.11907v2)
# CVPR 2020

# GhostNetV2: Enhance Cheap Operation with Long-Range Attention (https://arxiv.org/abs/2211.12905)
# NeurIPS 2022

class GhostConv2d(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]] = 1, 
                        cheap_kernel_size: Union[int, Tuple[int, int]] = 5, 
                        stride: int = 1, 
                        groups: int = 1, 
                        bias: bool = False, 
                        padding_mode: str = 'zeros',
                        bn_kwargs: dict = None,
                        use_nonlinear: bool = True):
        super(GhostConv2d, self).__init__()
        internal_channels  = out_channels // 2  # hidden channels
        kernel_size = _pair(kernel_size)
        cheap_kernel_size = _pair(cheap_kernel_size)

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        self.primary_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(  in_channels=in_channels,
                                    out_channels=internal_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size[0]//2, kernel_size[1]//2),
                                    dilation=1,
                                    groups=groups,
                                    bias=bias,
                                    padding_mode=padding_mode)),
            ('bn', torch.nn.BatchNorm2d(num_features=internal_channels, **bn_kwargs)),
            ('act', nn.ReLU(inplace=True) if use_nonlinear else nn.Identity() ),

        ]))

        self.cheap_operation = nn.Sequential(OrderedDict([
            ('dw_conv', nn.Conv2d(  in_channels=internal_channels,
                                    out_channels=internal_channels,
                                    kernel_size=cheap_kernel_size,
                                    stride=1,
                                    padding=(cheap_kernel_size[0]//2, cheap_kernel_size[1]//2),
                                    dilation=1,
                                    groups=internal_channels,
                                    bias=bias,
                                    padding_mode=padding_mode)),
            ('dw_bn', torch.nn.BatchNorm2d(num_features=internal_channels, **bn_kwargs)),
            ('dw_act', nn.ReLU(inplace=True) if use_nonlinear else nn.Identity() ),

        ]))

    def forward(self, x):
        y = self.primary_conv(x)
        return torch.cat([y, self.cheap_operation(y)], dim=1)
    
class Ghostv2Conv2d(nn.Module):

    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]] = 1, 
                        cheap_kernel_size: Union[int, Tuple[int, int]] = 5, 
                        stride: int = 1, 
                        groups: int = 1, 
                        bias: bool = False, 
                        padding_mode: str = 'zeros',
                        bn_kwargs: dict = None,
                        use_nonlinear: bool = True):
        super(Ghostv2Conv2d, self).__init__()

        kernel_size = _pair(kernel_size)

        self.ghost_conv = GhostConv2d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = kernel_size, 
                        cheap_kernel_size = cheap_kernel_size, 
                        stride = stride, 
                        groups = groups, 
                        bias = bias, 
                        padding_mode = padding_mode,
                        bn_kwargs = bn_kwargs,
                        use_nonlinear =  use_nonlinear)
        
        self.dfc_attn = nn.Sequential( 
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBN(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False),
            ConvBN(out_channels, out_channels, kernel_size=(1,5), stride=1, padding=(0,2), groups=groups, bias=False),
            ConvBN(out_channels, out_channels, kernel_size=(5,1), stride=1, padding=(2,0), groups=groups, bias=False),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        y1 = self.ghost_conv(x)
        y2 = self.dfc_attn(x)
        return y1*F.interpolate(y2, size=y1.shape[-2:], mode='nearest')
    

