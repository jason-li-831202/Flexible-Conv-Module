import torch
from torch import nn
from typing import Union, Tuple, Type
from collections import OrderedDict

from Conv.Basic.common import _pair
try :
    from Attention.SEModule import SEBlock
except :
    import sys
    sys.path.append("..")
    from Attention.SEModule import SEBlock

__all__ = ["DepthwiseSeparableConv2d"]

# Improve DW Conv: 
# PP-LCNet: A Lightweight CPU Convolutional Neural Network (https://arxiv.org/pdf/2109.15099)
# CVPR 2021

class DepthwiseSeparableConv2d(nn.Module):
    """ Residual Block
        Conv kxk -> (SE Attention) -> Conv 1x1
    """
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]], 
                        stride: int = 1, 
                        padding: Union[int, Tuple[int, int]] = 0, 
                        dilation: int = 1, 
                        bias: bool = False, 
                        padding_mode: str = 'zeros',
                        use_se: bool = False,
                        use_bn: Union[bool, Tuple[bool, bool]] = True,
                        use_nonlinear: Union[bool, Tuple[bool, bool]] = True,
                        bn_kwargs: dict = None,
                        activation: Type[nn.Module] = nn.ReLU(inplace=True)):
        super().__init__()
        use_bn = _pair(use_bn)
        use_nonlinear = _pair(use_nonlinear)
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        self.dw_part = nn.Sequential(OrderedDict([
            ('dw_conv', nn.Conv2d(  in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_channels,
                                    bias=bias,
                                    padding_mode=padding_mode)),
            ('dw_bn', torch.nn.BatchNorm2d(num_features=in_channels, **bn_kwargs) if use_bn[0] else nn.Identity() ),
            ('dw_act', activation if use_nonlinear[0] else nn.Identity() ),

        ]))

        self.se = SEBlock(in_channels, internal_reduce=16) if use_se else nn.Sequential()

        self.pw_part = nn.Sequential(OrderedDict([
            ('pw_conv', nn.Conv2d(  in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=bias)),
            ('pw_bn', torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs) if use_bn[1] else nn.Identity() ),
            ('pw_act', activation if use_nonlinear[1] else nn.Identity() ),

        ]))

    def forward(self, x):
        return self.pw_part(self.se(self.dw_part(x)))
